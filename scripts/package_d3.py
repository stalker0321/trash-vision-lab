"""Package verified D3 models and evaluation evidence, without dataset images."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from summarize_d3 import collect


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nano-run', type=Path, required=True)
    parser.add_argument('--medium-run', type=Path, required=True)
    parser.add_argument('--nano-export', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + '.sha256').exists():
        raise FileExistsError('Refusing to overwrite a release archive or its checksum')
    manifest = read(ROOT / 'manifests/d3.json')
    queue = read(ROOT / 'configs/d3_evaluation.json')
    rows = collect(args.report / 'inputs', queue, manifest)
    entries = {}
    runs = []

    def add(name, path):
        if name in entries:
            raise ValueError('Duplicate archive entry: ' + name)
        entries[name] = path.read_bytes()

    for name, model, run_root in [('nano', 'yolo26n', args.nano_run),
                                  ('medium', 'yolo26m', args.medium_run)]:
        run = read(run_root / 'run.json')
        if (run['status'] != 'complete' or run['smoke'] or run['model'] != model or
                run['seed'] != 42 or run['dataset_fingerprint'] != manifest['fingerprint']):
            raise ValueError('Not a completed D3 seed42 training run: ' + name)
        with (run_root / 'results.csv').open() as stream:
            epochs = [int(row['epoch']) for row in csv.DictReader(stream)]
        if epochs != list(range(1, 151)):
            raise ValueError('Expected all 150 training epochs: ' + name)
        weights = run_root / 'weights/best.pt'
        if sha(weights) != run['best_sha256']:
            raise ValueError('Checkpoint checksum mismatch: ' + name)
        for row in rows:
            if row['name'] == name + '_d3' and row['weights_sha256'] != run['best_sha256']:
                raise ValueError('Report evaluates another checkpoint: ' + name)
        add('models/' + name + '-d3-best.pt', weights)
        add('training/' + name + '.csv', run_root / 'results.csv')
        safe_keys = ['model', 'seed', 'dataset_fingerprint', 'pretrained_sha256', 'protocol',
                     'signature', 'status', 'best_sha256', 'gpu', 'python', 'torch',
                     'ultralytics', 'elapsed_seconds']
        runs.append(dict(name=name, **{key: run[key] for key in safe_keys}))

    export = read(args.nano_export / 'export.json')
    if (export['status'] != 'complete' or export['weights_sha256'] != runs[0]['best_sha256'] or
            export['dataset_fingerprint'] != manifest['fingerprint'] or
            export['target'] != 'rv1103' or export['imgsz'] != 320 or export['quantize'] != 8):
        raise ValueError('Export does not match the completed D3 Nano checkpoint')
    for source, name, key in [('nano.onnx', 'nano-d3.onnx', 'onnx_sha256'),
                              ('model/nano-rv1103.rknn', 'nano-d3-rv1103.rknn', 'rknn_sha256')]:
        path = args.nano_export / source
        if sha(path) != export[key]:
            raise ValueError('Export checksum mismatch: ' + source)
        add('models/' + name, path)
    receipt = {k: v for k, v in export.items() if k not in ['calibration_ids', 'calibration_copies']}
    receipt['calibration_images'] = len(export['calibration_ids'])
    entries['export.json'] = (json.dumps(receipt, indent=2) + '\n').encode()
    entries['training/runs.json'] = (json.dumps(runs, indent=2) + '\n').encode()
    for row in rows:
        name = row['name'] + '_' + row['split']
        add('evaluation/' + name + '/summary.json', args.report / 'inputs' / name / 'summary.json')
    for name in ['summary.json', 'metrics.csv', 'per-class.csv', 'README.md', 'comparison.svg', 'comparison.png']:
        add('evaluation/' + name, args.report / name)
    for name in ['d3_training.json', 'd3_evaluation.json', 'requirements-training-linux.lock', 'requirements-rknn.lock']:
        add('configs/' + name, ROOT / 'configs' / name)
    add('manifests/d3.json', ROOT / 'manifests/d3.json')
    lines = ['# Trash Vision Lab — D3 model artifacts', '',
             'Both models completed 150 epochs from pretrained initialization, seed 42.',
             'Nano uses 320 px; Medium uses 640 px. Checkpoints are selected on validation.', '',
             '| Model | D2 test AP50–95 | D3 test AP50–95 |', '| --- | ---: | ---: |']
    lookup = {(r['name'], r['split']): r for r in rows}
    for name in ['nano', 'medium']:
        values = [100 * lookup[name + '_' + version, 'test']['ap50_95'] for version in ['d2', 'd3']]
        lines.append(f'| {name} | {values[0]:.2f}% | {values[1]:.2f}% |')
    lines += ['', 'The test uses the same reviewed labels for D2/D3; historical D2 best weights',
              'were selected on earlier validation labels. This is one seed and a retrospective',
              'holdout; the hard subset is part of test. See evaluation/README.md.', '',
              'Classes: plastic, paper, metal, glass. These are visual annotation conventions.',
              'The Nano ONNX output is normalized cxcywh plus four class scores, 1×8×2100.',
              'It has no separate objectness channel. The RV1103 RKNN model is INT8.',
              'D3 INT8 hardware accuracy/performance is pending; historical D2 board timings',
              'do not describe this artifact. Source images are not included.', '',
              'Repository: https://github.com/stalker0321/trash-vision-lab',
              'Every payload checksum is recorded in artifacts.json.', '']
    entries['README.md'] = '\n'.join(lines).encode()
    checks = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
              for name, data in sorted(entries.items())}
    entries['artifacts.json'] = (json.dumps({'dataset_fingerprint': manifest['fingerprint'],
                                            'files': checks}, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    with temporary.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for name, data in sorted(entries.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    with tarfile.open(temporary, 'r:gz') as archive:
        if set(archive.getnames()) != set(entries):
            raise ValueError('Archive members differ from intended payload')
        for name, expected in checks.items():
            data = archive.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Archive payload checksum mismatch: ' + name)
    temporary.replace(args.output)
    checksum = sha(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(checksum + '  ' + args.output.name + '\n')
    print(json.dumps({'archive': str(args.output), 'sha256': checksum, 'files': len(entries),
                      'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
