"""Freeze and materialize D3 from reviewed labels and the unchanged D2 split.

Source pools are read-only. Every output image is an independent copy; an
existing version is never overwritten. Only Python's standard library is used.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def pool_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes source pool: {relative}')
    return path


def normalize_labels(path):
    """Use the existing D2 clipping convention, without a size threshold."""
    rows, clipped = [], 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{path}:{number}: invalid YOLO row')
        cls, x, y, w, h = values
        if cls not in range(4) or not 0 <= x <= 1 or not 0 <= y <= 1 or min(w, h) <= 0:
            raise ValueError(f'{path}:{number}: invalid class/geometry')
        left, top, right, bottom = max(0, x-w/2), max(0, y-h/2), min(1, x+w/2), min(1, y+h/2)
        clipped += left != x-w/2 or top != y-h/2 or right != x+w/2 or bottom != y+h/2
        rows.append([int(cls), (left+right)/2, (top+bottom)/2, right-left, bottom-top])
    text = ''.join(str(row[0]) + ' ' + ' '.join(f'{v:.9f}' for v in row[1:]) + '\n' for row in rows)
    return text, [sum(r[0] == c for r in rows) for c in range(4)], clipped


def validate_splits(manifest):
    real_sources = {'taco_official', 'taco_community', 'my_photos'}
    index = {r['id']: r for r in manifest['images']}
    if len(index) != len(manifest['images']):
        raise ValueError('Duplicate image IDs')
    groups, hashes = {}, {}
    for split in ['d3_train', 'val', 'test']:
        ids = manifest['splits'][split]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {split}')
        for key in ids:
            row = index[key]
            if split != 'd3_train' and row['source'] not in real_sources:
                raise ValueError(f'Non-real holdout: {key}')
            for seen, value in [(groups, row['group_id']), (hashes, row['image_sha256'])]:
                if value in seen and seen[value] != split:
                    raise ValueError(f'Cross-split leakage: {key}')
                seen[value] = split
    if not set(manifest['splits']['hard_test']) <= set(manifest['splits']['test']):
        raise ValueError('Hard test must be a test subset')
    for key in manifest['splits']['calibration']:
        if key not in manifest['splits']['d3_train'] or index[key]['source'] not in real_sources:
            raise ValueError(f'Calibration must be real train: {key}')


def freeze(baseline_root, reviewed_root):
    old_path = baseline_root / 'manifests/dataset.json'
    old = json.loads(old_path.read_text())
    previous = dict(old)
    if digest({k: v for k, v in old.items() if k != 'fingerprint'}) != old['fingerprint']:
        raise ValueError('Invalid baseline fingerprint')
    with (reviewed_root / 'manifests/manifest_final.csv').open() as stream:
        source_rows = list(csv.DictReader(stream))
    source_index = {r['image_id']: r for r in source_rows}
    if len(source_index) != len(source_rows):
        raise ValueError('Duplicate reviewed source IDs')
    splits = {('d3_train' if k == 'd2_train' else k): v for k, v in old['splits'].items()
              if k not in {'d0_train', 'd1_train'}}
    selected = set().union(*(set(splits[k]) for k in ['d3_train', 'val', 'test']))
    records, clipping, changed = [], 0, set()
    for old_row in old['images']:
        if old_row['id'] not in selected:
            continue
        source = source_index[old_row['id'].removeprefix('clean:')]
        image_path = pool_path(reviewed_root, source['file_path'])
        if file_hash(image_path) != old_row['source_sha256']:
            raise ValueError(f'Reviewed image content changed: {old_row["id"]}')
        label_path = pool_path(reviewed_root, source['label_path'])
        text, counts, clipped = normalize_labels(label_path)
        clipping += clipped
        row = dict(old_row, class_counts=counts, bbox_count=sum(counts),
                   label_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   reviewed_image=source['file_path'], reviewed_label=source['label_path'],
                   reviewed_label_sha256=file_hash(label_path))
        old_label_path = pool_path(baseline_root / 'data/prepared', old_row['label'])
        if file_hash(old_label_path) != old_row['label_sha256']:
            raise ValueError(f'Baseline label changed: {old_row["id"]}')
        def numeric_rows(content):
            return sorted(tuple(round(float(v), 8) for v in l.split()) for l in content.splitlines() if l.strip())
        if numeric_rows(text) != numeric_rows(old_label_path.read_text()):
            changed.add(row['id'])
        records.append(row)
    result = {'schema_version': 1, 'version': 'd3', 'names': old['names'],
              'baseline_fingerprint': previous['fingerprint'],
              'reviewed_manifest_sha256': file_hash(reviewed_root / 'manifests/manifest_final.csv'),
              'policy': 'D2 image IDs/groups/quotas, reviewed-v1 labels; same clipping; real-only holdout.',
              'images': sorted(records, key=lambda r: r['id']), 'splits': splits}
    validate_splits(result)
    result['fingerprint'] = digest(result)
    index = {r['id']: r for r in records}
    summary = {'version': 'd3', 'fingerprint': result['fingerprint'], 'clipped_boxes': clipping, 'splits': {}}
    for split in ['d3_train', 'val', 'test', 'hard_test', 'calibration']:
        ids = splits[split]
        summary['splits'][split] = {'images': len(ids), 'boxes': sum(index[i]['bbox_count'] for i in ids),
            'empty': sum(index[i]['bbox_count'] == 0 for i in ids), 'changed_label_images': len(set(ids) & changed),
            'classes': {name: sum(index[i]['class_counts'][c] for i in ids) for c, name in enumerate(result['names'])},
            'sources': dict(collections.Counter(index[i]['source'] for i in ids))}
    return result, summary


def materialize(manifest, baseline_root, reviewed_root, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing dataset: {output}')
    output.mkdir(parents=True)
    for row in manifest['images']:
        source = pool_path(reviewed_root, row['reviewed_image'])
        source_hash = file_hash(source)
        if source_hash != row['source_sha256']:
            raise ValueError(f'Reviewed image content changed: {row["id"]}')
        if source_hash != row['image_sha256']:
            if baseline_root is None:
                raise ValueError(f'An intact oriented image is required from --baseline-root: {row["id"]}')
            source = pool_path(baseline_root / 'data/prepared', row['image'])
            if file_hash(source) != row['image_sha256']:
                raise ValueError(f'No intact oriented image available: {row["id"]}')
        destination = pool_path(output, row['image'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        label_source = pool_path(reviewed_root, row['reviewed_label'])
        if file_hash(label_source) != row['reviewed_label_sha256']:
            raise ValueError(f'Reviewed label changed during build: {row["id"]}')
        text, _, _ = normalize_labels(label_source)
        label = pool_path(output, row['label'])
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(text)
        if file_hash(destination) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
            raise ValueError(f'Output hash mismatch: {row["id"]}')
    index = {r['id']: r for r in manifest['images']}
    for split, ids in manifest['splits'].items():
        path = output / 'splits' / f'{split}.txt'
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(str((output / index[i]['image']).resolve()) + '\n' for i in ids))
    config = {'path': str(output.resolve()), 'train': 'splits/d3_train.txt', 'val': 'splits/val.txt',
              'test': 'splits/test.txt', 'names': manifest['names']}
    # JSON is valid YAML and keeps paths/platform escaping unambiguous.
    write_json(output / 'data.yaml', config)
    write_json(output / 'dataset.json', manifest)


def read_frozen_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 'd3' or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported frozen manifest')
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest.get('fingerprint'):
        raise ValueError('Invalid frozen manifest fingerprint')
    validate_splits(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--reviewed-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('manifests/d3.json'))
    parser.add_argument('--output', type=Path, default=Path('data/versions/d3'))
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--from-manifest', action='store_true',
                        help='Materialize the frozen manifest without reconstructing historical D2 labels')
    args = parser.parse_args()
    baseline = args.baseline_root.resolve() if args.baseline_root else None
    if args.from_manifest:
        result = read_frozen_manifest(args.manifest)
        index = {row['id']: row for row in result['images']}
        summary = {'version': 'd3', 'fingerprint': result['fingerprint'], 'mode': 'frozen_manifest',
                   'splits': {name: {'images': len(ids), 'boxes': sum(index[i]['bbox_count'] for i in ids)}
                              for name, ids in result['splits'].items()}}
    else:
        if baseline is None:
            parser.error('--baseline-root is required when freezing a new manifest')
        result, summary = freeze(baseline, args.reviewed_root.resolve())
        if args.manifest.exists() and json.loads(args.manifest.read_text()) != result:
            raise ValueError('Frozen D3 manifest differs; use a new version/path')
        write_json(args.manifest, result)
    if not args.plan_only:
        materialize(result, baseline, args.reviewed_root, args.output.resolve())
        write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
