"""Reproduce a train-only label-deletion ablation without editing frozen D3."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from datasets.build_d3 import digest, file_hash, pool_path, read_frozen_manifest, validate_splits, write_json


def filtered_label(raw, width, height, threshold, reference_size):
    kept, removed, counts = [], [], [0]*4
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid YOLO label')
        cls, x, y, w, h = values
        if cls not in range(4) or min(w, h) <= 0:
            raise ValueError('Invalid class or box dimensions')
        short_side = min(w*width, h*height) * reference_size / max(width, height)
        if short_side < threshold:
            removed.append({'line_number': number, 'yolo': line, 'projected_short_side': short_side})
        else:
            kept.append(line)
            counts[int(cls)] += 1
    # Preserve original bytes whenever no box is removed, including holdout.
    result = (''.join(line + '\n' for line in kept)).encode() if removed else raw
    return result, counts, removed


def freeze(source, threshold=8.0, reference_size=320):
    if not math.isfinite(threshold) or threshold <= 0 or reference_size <= 0:
        raise ValueError('Size parameters must be positive')
    parent = read_frozen_manifest(source / 'dataset.json')
    variant = copy.deepcopy(parent)
    variant.pop('fingerprint')
    variant.update(version='d3_drop_lt8', parent_fingerprint=parent['fingerprint'],
        policy={'operation': 'omit_train_labels_below_projected_short_side',
                'threshold_px': threshold, 'reference_long_side_px': reference_size,
                'comparison': 'strict_less_than', 'split': 'd3_train',
                'images': 'unchanged_independent_copies', 'holdout': 'unchanged_bytes',
                'note': 'Omitted objects remain in images; no ignore-region loss.'})
    train = set(parent['splits']['d3_train'])
    changes, label_bytes = [], {}
    for row in variant['images']:
        raw = pool_path(source, row['label']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['label_sha256']:
            raise ValueError(f'Source label hash mismatch: {row["id"]}')
        if row['id'] in train:
            result, counts, removed = filtered_label(raw, row['width'], row['height'], threshold, reference_size)
            if removed:
                changes.append({'id': row['id'], 'source_label_sha256': row['label_sha256'], 'removed': removed})
                row.update(label_sha256=hashlib.sha256(result).hexdigest(), class_counts=counts, bbox_count=sum(counts))
            label_bytes[row['id']] = result
        else:
            label_bytes[row['id']] = raw
    variant['fingerprint'] = digest(variant)
    validate_splits(variant)
    report = {'parent_fingerprint': parent['fingerprint'], 'fingerprint': variant['fingerprint'],
              'policy': variant['policy'], 'changed_images': len(changes),
              'removed_boxes': sum(len(c['removed']) for c in changes), 'changes': changes,
              'train_boxes': sum(r['bbox_count'] for r in variant['images'] if r['id'] in train)}
    return variant, label_bytes, report


def materialize(source, output, manifest, labels):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    for row in manifest['images']:
        image = pool_path(source, row['image'])
        if file_hash(image) != row['image_sha256']:
            raise ValueError(f'Source image hash mismatch: {row["id"]}')
        destination = pool_path(output, row['image'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, destination)
        target_label = pool_path(output, row['label'])
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_bytes(labels[row['id']])
        if file_hash(destination) != row['image_sha256'] or file_hash(target_label) != row['label_sha256']:
            raise ValueError(f'Copied data hash mismatch: {row["id"]}')
    write_json(output / 'dataset.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/versions/d3'))
    parser.add_argument('--manifest', type=Path, default=Path('manifests/d3-drop-lt8.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest, labels, report = freeze(args.source)
    if args.manifest.exists():
        if json.loads(args.manifest.read_text()) != manifest:
            raise ValueError('Variant differs from the frozen manifest')
    else:
        write_json(args.manifest, manifest)
    if args.output:
        materialize(args.source, args.output, manifest, labels)
    write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()
