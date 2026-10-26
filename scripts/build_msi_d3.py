"""Copy unchanged canonical images into D3; refuse any missing/hash-mismatched file."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.build_d3 import digest, file_hash, pool_path, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root', type=Path, required=True)
    args = parser.parse_args()
    data = ROOT / 'data/versions/d3'
    manifest = json.loads((data / 'dataset.json').read_text())
    assert digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) == manifest['fingerprint']
    validate_splits(manifest)
    missing = []
    for n, row in enumerate(manifest['images'], 1):
        destination = pool_path(data, row['image'])
        if destination.exists():
            if file_hash(destination) != row['image_sha256']:
                raise ValueError(f'Existing output differs: {row["id"]}')
        else:
            source = pool_path(args.canonical_root, row['reviewed_image'])
            if not source.is_file() or file_hash(source) != row['image_sha256']:
                missing.append({'id': row['id'], 'image': row['image'], 'sha256': row['image_sha256']})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_hash(destination) != row['image_sha256']:
                raise ValueError(f'Copy mismatch: {row["id"]}')
        if file_hash(pool_path(data, row['label'])) != row['label_sha256']:
            raise ValueError(f'Label mismatch: {row["id"]}')
        if n % 500 == 0:
            print(f'Checked {n}/{len(manifest["images"])} images', flush=True)
    (ROOT / 'missing-images.json').write_text(json.dumps(missing, indent=2) + '\n')
    if missing:
        raise SystemExit(f'{len(missing)} images require transfer from the frozen D3; see missing-images.json')
    print('D3 image and label hashes verified:', manifest['fingerprint'])


if __name__ == '__main__':
    main()
