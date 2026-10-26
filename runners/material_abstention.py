"""Test material-label abstention on fixed boxes; choose rules on val, then evaluate test once."""
import argparse
import json
import os
from pathlib import Path

from datasets.build_d3 import digest, file_hash, pool_path, validate_splits, write_json


def summarize(frames, method, threshold, names):
    records = [d for frame in frames for d in frame['detections']]
    matched = [d for d in records if d['true_class'] is not None]
    kept = lambda d: d[method] >= threshold
    accepted = [d for d in records if kept(d)]
    accepted_matched = [d for d in matched if kept(d)]
    correct = lambda d: d['true_class'] is not None and d['predicted_class'] == d['true_class']
    ratio = lambda a, b: a / b if b else None
    return dict(method=method, threshold=threshold, images=len(frames),
                ground_truth_boxes=sum(f['ground_truth_boxes'] for f in frames),
                detections=len(records), matched_boxes=len(matched),
                unmatched_detections=len(records)-len(matched),
                named_detections=len(accepted), unknown_material_detections=len(records)-len(accepted),
                named_matched_boxes=len(accepted_matched),
                matched_box_material_coverage=ratio(len(accepted_matched), len(matched)),
                named_matched_material_accuracy=ratio(sum(map(correct, accepted_matched)), len(accepted_matched)),
                named_detection_precision=ratio(sum(map(correct, accepted)), len(accepted)),
                correct_labels_deferred=sum(correct(d) and not kept(d) for d in matched),
                wrong_labels_deferred=sum(not correct(d) and not kept(d) for d in matched),
                named_unmatched_detections=sum(d['true_class'] is None for d in accepted),
                per_true_class={name: dict(matched=sum(d['true_class'] == c for d in matched),
                    named=sum(d['true_class'] == c for d in accepted_matched),
                    correct_named=sum(d['true_class'] == c and correct(d) for d in accepted_matched))
                    for c, name in enumerate(names)})


def select_rule(candidates):
    eligible = [r for r in candidates if r['named_matched_boxes'] >= 50 and
                (r['matched_box_material_coverage'] or 0) >= .9]
    if not eligible:
        raise ValueError('Insufficient matched validation boxes for the predefined selection rule')
    return max(eligible, key=lambda r: (r['named_matched_material_accuracy'],
                                       r['matched_box_material_coverage'], -r['threshold']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.resolve()
    manifest = json.loads((data / 'dataset.json').read_text())
    export = json.loads((args.export_root / 'export.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('Invalid manifest fingerprint')
    if export['status'] != 'complete' or export['dataset_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Export and dataset provenance differ')
    validate_splits(manifest)
    graph = args.export_root / 'nano.onnx'
    if file_hash(graph) != export['onnx_sha256']:
        raise ValueError('Export graph changed')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    output.mkdir(parents=True)
    os.environ['OMP_NUM_THREADS'] = '2'
    import cv2
    import numpy as np
    import onnxruntime as ort
    cv2.setNumThreads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 2
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    index = {r['id']: r for r in manifest['images']}

    def iou(box, others):
        if not len(others):
            return np.empty(0)
        wh = np.maximum(0, np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2]))
        intersection = wh.prod(axis=1)
        area = np.maximum(0, box[2:]-box[:2]).prod()
        other_area = np.maximum(0, others[:, 2:]-others[:, :2]).prod(axis=1)
        return intersection / np.maximum(area + other_area - intersection, 1e-9)

    def predict(split):
        frames = []
        with (output / (split + '-predictions.jsonl')).open('w') as stream:
            for number, image_id in enumerate(manifest['splits'][split], 1):
                row = index[image_id]
                for kind in ['image', 'label']:
                    if file_hash(pool_path(data, row[kind])) != row[kind + '_sha256']:
                        raise ValueError(f'Changed {kind}: {image_id}')
                image = cv2.imread(str(pool_path(data, row['image'])), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot read image: {image_id}')
                height, width = image.shape[:2]
                gain = min(320 / width, 320 / height)
                rw, rh = round(width * gain), round(height * gain)
                left, top = round((320-rw)/2-.1), round((320-rh)/2-.1)
                boxed = cv2.copyMakeBorder(cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR),
                    top, 320-rh-top, left, 320-rw-left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                tensor = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
                raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                if raw.shape != (1, 8, 2100) or not np.isfinite(raw).all():
                    raise ValueError('Unexpected normalized output')
                scores = raw[0, 4:, :].T
                classes, confidence = scores.argmax(axis=1), scores.max(axis=1)
                margins = confidence - np.sort(scores, axis=1)[:, -2]
                xywh = raw[0, :4, :].T * 320
                boxes = np.concatenate([xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2], axis=1)
                candidates = np.flatnonzero((confidence >= .25) & (xywh[:, 2:] > 0).all(axis=1))
                candidates = candidates[np.argsort(-confidence[candidates], kind='stable')]
                retained = []
                for anchor in candidates:
                    same_class = [k for k in retained if classes[k] == classes[anchor]]
                    if len(same_class) and (iou(boxes[anchor], boxes[same_class]) > .45).any():
                        continue
                    retained.append(int(anchor))
                    if len(retained) == 300:
                        break
                labels = [[float(v) for v in line.split()] for line in pool_path(data, row['label']).read_text().splitlines() if line.strip()]
                if len(labels) != row['bbox_count']:
                    raise ValueError('Label count differs from manifest')
                targets = np.asarray(labels, dtype=np.float64).reshape(-1, 5)
                gt_boxes = np.empty((len(targets), 4))
                gt_boxes[:, :2] = (targets[:, 1:3] - targets[:, 3:5]/2) * [width, height]
                gt_boxes[:, 2:] = (targets[:, 1:3] + targets[:, 3:5]/2) * [width, height]
                used, detections = set(), []
                for anchor in retained:  # confidence-ordered, class-agnostic one-to-one IoU matching
                    box = (boxes[anchor] - [left, top, left, top]) / gain
                    box = np.clip(box, 0, [width, height, width, height])
                    overlaps = iou(box, gt_boxes)
                    for taken in used:
                        overlaps[taken] = -1
                    matched = int(overlaps.argmax()) if len(overlaps) and overlaps.max() >= .5 else None
                    if matched is not None:
                        used.add(matched)
                    detections.append(dict(predicted_class=int(classes[anchor]), confidence=float(confidence[anchor]),
                        margin=float(margins[anchor]), true_class=int(targets[matched, 0]) if matched is not None else None,
                        matched_gt_index=matched, xyxy=box.tolist()))
                frame = dict(id=image_id, ground_truth_boxes=len(labels), detections=detections)
                frames.append(frame)
                stream.write(json.dumps(frame) + '\n')
                if number % 100 == 0:
                    print(f'{split}: {number}/{len(manifest["splits"][split])}', flush=True)
        return frames

    grids = {'margin': [0, .025, .05, .075, .10, .15, .20, .25],
             'confidence': [.25, .30, .35, .40, .45, .50, .60, .70]}
    protocol = dict(base_confidence=.25, classwise_nms_iou=.45, match_iou=.5, max_det=300,
        matching='Confidence-ordered class-agnostic greedy one-to-one matching',
        threshold_grids=grids, selection='Maximize named matched material accuracy on val, retaining >=90% matched boxes and >=50 named matches; tie: more coverage, then lower threshold.',
        inference='Nano320 normalized FP32 ONNX, one-to-many branch with explicit confidence/NMS; not the AP evaluation protocol',
        action='Keep every detected box. Replace only its material label by unknown when the rule rejects it.')
    write_json(output / 'protocol.json', protocol)
    validation = predict('val')
    candidates = {method: [summarize(validation, method, t, manifest['names']) for t in values] for method, values in grids.items()}
    selected = {method: select_rule(rows) for method, rows in candidates.items()}
    # Lock the chosen rules on disk before producing any test predictions.
    write_json(output / 'selection.json', dict(dataset_fingerprint=manifest['fingerprint'],
        weights_sha256=export['weights_sha256'], selected=selected, selection_split='val'))
    test = predict('test')
    test_results = {method: summarize(test, method, row['threshold'], manifest['names']) for method, row in selected.items()}
    baseline = summarize(test, 'margin', 0, manifest['names'])
    for image_id in manifest['splits']['val'] + manifest['splits']['test']:
        row = index[image_id]
        for kind in ['image', 'label']:
            if file_hash(pool_path(data, row[kind])) != row[kind + '_sha256']:
                raise ValueError('Experiment changed source data')
    if file_hash(graph) != export['onnx_sha256']:
        raise ValueError('Experiment changed the export graph')
    report = dict(status='complete', experimental=True, weights_sha256=export['weights_sha256'],
        onnx_sha256=export['onnx_sha256'], selection_sha256=file_hash(output / 'selection.json'),
        dataset_fingerprint=manifest['fingerprint'], protocol=protocol,
        validation_candidates=candidates, selected_on_validation=selected, test_baseline=baseline, test_selected=test_results,
        caveats=['Class scores are not calibrated probabilities or a separate objectness score.',
                 'Abstention changes material naming, not box recall or the underlying false detections.',
                 'Only known four-material ground truth is evaluated; this is not open-set unknown-object detection.',
                 'FP32 host inference is not an INT8 board measurement; thresholds need board verification.',
                 'One checkpoint and a retrospectively reviewed holdout; no claim of statistical significance.'])
    write_json(output / 'summary.json', report)
    print(json.dumps(dict(selected={k:v['threshold'] for k,v in selected.items()}, test=test_results)), flush=True)


if __name__ == '__main__':
    main()
