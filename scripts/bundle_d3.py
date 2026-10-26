"""Build an allowlisted D3 transport archive, checking every payload hash."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.build_d3 import digest, file_hash, pool_path, validate_splits


class CheckedReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    data = ROOT / 'data/versions/d3'
    manifest = json.loads((data / 'dataset.json').read_text())
    recipe = json.loads((ROOT / 'configs/d3_training.json').read_text())
    assert digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) == manifest['fingerprint']
    assert manifest['fingerprint'] == recipe['dataset_fingerprint']
    validate_splits(manifest)
    files = {}
    for name in ['datasets/__init__.py', 'datasets/build_d3.py', 'runners/train_d3.py',
                 'configs/d3_training.json', 'configs/requirements-training-linux.lock',
                 'scripts/bootstrap_training.py', 'scripts/train_job.py', 'scripts/build_msi_d3.py']:
        files[name] = (ROOT / name, file_hash(ROOT / name))
    files['data/versions/d3/dataset.json'] = (data / 'dataset.json', file_hash(data / 'dataset.json'))
    for model, spec in recipe['models'].items():
        files[f'weights/{model}.pt'] = (args.weights_root / (model + '.pt'), spec['pretrained_sha256'])
    for row in manifest['images']:
        for kind in (['label'] if args.metadata_only else ['image', 'label']):
            files['data/versions/d3/' + row[kind]] = (pool_path(data, row[kind]), row[kind + '_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(args.output.suffix + '.part')
    packed = {}
    try:
        with temporary.open('xb') as output, tarfile.open(fileobj=output, mode='w|') as archive:
            for name, (path, expected) in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open('rb') as source:
                    reader = CheckedReader(source)
                    archive.addfile(info, reader)
                if reader.sha.hexdigest() != expected:
                    raise ValueError(f'Payload changed or mismatched: {name}')
                packed[name] = expected
            summary = {'dataset_fingerprint': manifest['fingerprint'], 'metadata_only': args.metadata_only, 'files': packed}
            body = (json.dumps(summary, indent=2) + '\n').encode()
            info = tarfile.TarInfo('bundle-manifest.json'); info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        temporary.rename(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {'file': args.output.name, 'bytes': args.output.stat().st_size,
               'sha256': file_hash(args.output), 'payload_files': len(packed),
               'dataset_fingerprint': manifest['fingerprint'], 'metadata_only': args.metadata_only}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
