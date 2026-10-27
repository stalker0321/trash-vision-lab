"""Audit projected box sizes and validation recall from saved YOLO predictions.

No model inference or source edits. Size is measured at a common 320px longest
image side, before augmentation. Recall uses confidence-ordered one-to-one
matching at IoU >= .5; it is a diagnostic, not AP or a COCO area metric.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from datasets.build_d3 import digest, file_hash, pool_path, validate_splits


EDGES = (4, 8, 16, 32)
BUCKETS = ('<4', '4–8', '8–16', '16–32', '>=32')


def bucket(value):
    return next((i for i, edge in enumerate(EDGES) if value < edge), len(EDGES))


def dimensions(box, width, height):
    scale = 320 / max(width, height)
    return box[3] * width * scale, box[4] * height * scale


def xyxy(box, width, height):
    _, x, y, w, h = box
    return ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def match(truth, predictions, class_aware):
    used = set()
    for pred in sorted(predictions, key=lambda p: -p['score']):
        choices = [(iou(t['xyxy'], pred['xyxy']), i) for i, t in enumerate(truth)
                   if i not in used and (not class_aware or t['class'] == pred['class'])]
        if choices:
            overlap, index = max(choices)
            if overlap >= .5:
                used.add(index)
    return used


def analyze(data, evaluations, baseline=None):
    manifest = json.loads((data / 'dataset.json').read_text())
    assert digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) == manifest['fingerprint']
    validate_splits(manifest)
    index = {r['id']: r for r in manifest['images']}
    output = {'dataset_fingerprint': manifest['fingerprint'],
              'projection': '320 / max(image_width, image_height), before augmentation',
              'buckets_px': list(BUCKETS), 'splits': {}, 'validation_recall': [],
              'limitations': ['Validation only; no test-based threshold selection.',
                              'Recall is confidence-ordered one-to-one IoU>=0.5 matching, not AP.',
                              'Class-agnostic and class-aware matching are performed independently.',
                              'Common 320px size buckets also apply to Medium640 for comparable object groups.',
                              'Augmentation changes training sizes; no size threshold proves invisibility.']}
    truth = {}
    for split in ['d3_train', 'val']:
        sizes = []
        boxes_per_image = []
        class_counts = Counter()
        for key in manifest['splits'][split]:
            row = index[key]
            label = pool_path(data, row['label'])
            assert file_hash(label) == row['label_sha256'], key
            boxes = [tuple(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            assert len(boxes) == row['bbox_count'], key
            boxes_per_image.append([])
            records = []
            for box in boxes:
                w, h = dimensions(box, row['width'], row['height'])
                assert w > 0 and h > 0 and int(box[0]) == box[0] and 0 <= box[0] < 4
                size = {'short_side': min(w, h), 'sqrt_area': math.sqrt(w*h)}
                sizes.append(size)
                boxes_per_image[-1].append(size)
                class_counts[int(box[0])] += 1
                records.append({'class': int(box[0]), 'xyxy': xyxy(box, row['width'], row['height']), **size})
            if split == 'val':
                name = Path(row['image']).stem
                assert name not in truth, 'Ambiguous prediction image stem'
                truth[name] = records
        counts = {metric: [sum(bucket(s[metric]) == i for s in sizes) for i in range(len(BUCKETS))]
                  for metric in ['short_side', 'sqrt_area']}
        thresholds = {}
        for metric in ['short_side', 'sqrt_area']:
            thresholds[metric] = {str(edge): {
                'boxes_below': sum(s[metric] < edge for s in sizes),
                'images_with_any_below': sum(any(s[metric] < edge for s in boxes) for boxes in boxes_per_image),
                'nonempty_images_with_all_below': sum(bool(boxes) and all(s[metric] < edge for s in boxes) for boxes in boxes_per_image)
            } for edge in EDGES}
        output['splits'][split] = {'images': len(boxes_per_image), 'boxes': len(sizes),
                                   'buckets': counts, 'thresholds': thresholds,
                                   'classes': {manifest['names'][i]: class_counts[i] for i in range(4)}}
    if baseline is not None:
        old = json.loads((baseline / 'manifests/dataset.json').read_text())
        assert digest({k: v for k, v in old.items() if k != 'fingerprint'}) == old['fingerprint']
        assert old['fingerprint'] == manifest['baseline_fingerprint']
        assert old['splits']['d2_train'] == manifest['splits']['d3_train']
        old_index = {r['id']: r for r in old['images']}
        counts = {metric: [0]*len(BUCKETS) for metric in ['short_side', 'sqrt_area']}
        total = 0
        for key in old['splits']['d2_train']:
            row = old_index[key]
            assert row['image_sha256'] == index[key]['image_sha256']
            label = pool_path(baseline / 'data/prepared', row['label'])
            assert file_hash(label) == row['label_sha256']
            boxes = [tuple(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            assert len(boxes) == row['bbox_count']
            total += len(boxes)
            for box in boxes:
                w, h = dimensions(box, row['width'], row['height'])
                counts['short_side'][bucket(min(w, h))] += 1
                counts['sqrt_area'][bucket(math.sqrt(w*h))] += 1
        output['baseline_train'] = {'fingerprint': old['fingerprint'], 'boxes': total, 'buckets': counts,
            'd3_minus_d2_by_bucket': {metric: [new-old for new, old in zip(output['splits']['d3_train']['buckets'][metric], values)]
                                     for metric, values in counts.items()},
            'note': 'Net count changes within size buckets, not tracked identities of added objects; resizing, replacement and removal can change these counts.'}
    for name in ['nano_d2', 'nano_d3', 'medium_d2', 'medium_d3']:
        directory = evaluations / (name + '_val')
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['split'] == 'val' and summary['dataset_fingerprint'] == manifest['fingerprint']
        assert summary['boxes'] == sum(len(v) for v in truth.values())
        # Ultralytics 8.4.143 DetectionValidator uses category IDs 1..nc in JSON.
        assert summary['ultralytics'] == '8.4.143'
        assert summary['class_names'] == manifest['names']
        path = directory / 'predictions.json'
        raw = json.loads(path.read_text())
        predictions = defaultdict(list)
        for p in raw:
            key = str(p['image_id'])
            assert key in truth and 1 <= p['category_id'] <= 4
            x, y, w, h = p['bbox']
            assert all(math.isfinite(x) for x in [x,y,w,h,p['score']]) and w >= 0 and h >= 0
            predictions[key].append({'xyxy': (x, y, x+w, y+h), 'class': p['category_id']-1, 'score': p['score']})
        for confidence in [.001, .25]:
            for class_aware in [False, True]:
                hits = {metric: [0]*len(BUCKETS) for metric in ['short_side', 'sqrt_area']}
                for key, boxes in truth.items():
                    matched = match(boxes, [p for p in predictions[key] if p['score'] >= confidence], class_aware)
                    for i in matched:
                        for metric in hits:
                            hits[metric][bucket(boxes[i][metric])] += 1
                output['validation_recall'].append({'model': name, 'confidence': confidence, 'class_aware': class_aware,
                    'matched': hits, 'weights_sha256': summary['weights_sha256'],
                    'predictions_sha256': file_hash(path), 'total_predictions': len(raw),
                    'predictions_at_threshold': sum(p['score'] >= confidence for p in raw),
                    'matched_total': sum(hits['short_side'])})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/versions/d3'))
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.data_root, args.evaluations, args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['splits'], indent=2))


if __name__ == '__main__':
    main()
