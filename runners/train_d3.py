"""Run the frozen D3 Nano or Medium recipe; --plan-only needs no ML packages."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

from datasets.build_d3 import digest, file_hash, pool_path, validate_splits, write_json


def verify_dataset(root, manifest):
    validate_splits(manifest)
    for row in manifest['images']:
        for kind in ['image', 'label']:
            if file_hash(pool_path(root, row[kind])) != row[kind + '_sha256']:
                raise ValueError(f'Dataset content changed: {row["id"]} / {kind}')


def readonly_image_check(path):
    # Preserve valid phone JPEG trailers instead of letting verification rewrite
    # frozen images. Same policy as the September training pipeline.
    from PIL import Image
    from ultralytics.data.utils import IMG_FORMATS, exif_size
    with Image.open(path) as image:
        fmt = image.format.lower()
        size = exif_size(image)
        image.verify()
    if min(size) <= 9 or fmt not in IMG_FORMATS:
        raise ValueError(f'Invalid image: {path}')
    if fmt in {'jpg', 'jpeg'}:
        with open(path, 'rb') as stream:
            stream.seek(-2, 2)
            needs_decode = stream.read() != b'\xff\xd9'
        if needs_decode:
            with Image.open(path) as image:
                image.load()
    return '', (size[1], size[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolo26n', 'yolo26m'], required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/versions/d3'))
    parser.add_argument('--recipe', type=Path, default=Path('configs/d3_training.json'))
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, default=Path('runs/d3'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    data = args.data_root.resolve()
    manifest = json.loads((data / 'dataset.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('Dataset manifest fingerprint is invalid')
    if manifest['fingerprint'] != recipe['dataset_fingerprint'] or manifest['version'] != 'd3':
        raise ValueError('Dataset differs from the frozen D3 recipe')
    validate_splits(manifest)
    weights = (args.weights_root / (args.model + '.pt')).resolve()
    model_recipe = recipe['models'][args.model]
    if file_hash(weights) != model_recipe['pretrained_sha256']:
        raise ValueError('Pretrained checkpoint differs from the September starting weights')
    protocol = dict(recipe['protocol'], imgsz=model_recipe['imgsz'])
    if args.smoke:
        protocol.update(epochs=1, workers=0, close_mosaic=0, save_period=1, plots=False)
    name = f'{args.model}_d3_s{recipe["seed"]}_wd001' + ('_smoke' if args.smoke else '')
    plan = {'name': name, 'model': args.model, 'seed': recipe['seed'], 'dataset_fingerprint': manifest['fingerprint'],
            'pretrained_sha256': model_recipe['pretrained_sha256'], 'protocol': protocol,
            'comparator': model_recipe['comparator'], 'required_environment': recipe['environment'],
            'image_verification': 'readonly-decode-v1', 'smoke': args.smoke}
    signature = digest(plan)
    if args.plan_only:
        print(json.dumps(dict(plan, signature=signature), indent=2))
        return
    if not args.device.isdigit():
        parser.error('Select one CUDA GPU ID; CPU training is intentionally disabled')
    if '.'.join(platform.python_version_tuple()[:2]) != recipe['environment']['python']:
        raise RuntimeError('Use the pinned Python 3.12 environment')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    runs = args.runs_root.resolve()
    for key, subdir in [('YOLO_CONFIG_DIR', 'ultralytics'), ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch')]:
        os.environ[key] = str(runs / '.cache' / subdir)
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    import ultralytics.data.utils
    if torch.__version__.split('+')[0] != recipe['environment']['torch'] or ultralytics.__version__ != recipe['environment']['ultralytics']:
        raise RuntimeError('Restore the pinned torch 2.8.0 / Ultralytics 8.4.143 environment first')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    run = runs / name
    metadata = run / 'run.json'
    if args.resume:
        old = json.loads(metadata.read_text())
        if old['signature'] != signature or old['status'] == 'complete' or not (run / 'weights/last.pt').exists():
            raise ValueError('No compatible interrupted run to resume')
    elif run.exists():
        raise FileExistsError(f'Refusing to overwrite {run}; use --resume for an interrupted run')
    verify_dataset(data, manifest)
    run.mkdir(parents=True, exist_ok=True)
    # Absolute split lists are regenerated per run, so a transferred bundle does
    # not retain paths from the machine where D3 was assembled.
    index = {r['id']: r for r in manifest['images']}
    for output_name, source_split in [('train', 'smoke_train' if args.smoke else 'd3_train'), ('val', 'smoke_val' if args.smoke else 'val')]:
        (run / f'{output_name}.txt').write_text(''.join(str(data / index[i]['image']) + '\n' for i in manifest['splits'][source_split]))
    write_json(run / 'data.yaml', {'path': str(data), 'train': str(run / 'train.txt'), 'val': str(run / 'val.txt'), 'names': manifest['names']})
    ultralytics.data.utils.check_image = readonly_image_check
    settings.update({'sync': False, 'weights_dir': str(args.weights_root.resolve())})
    info = dict(plan, signature=signature, status='running', started_unix=time.time(),
                gpu=torch.cuda.get_device_name(0), platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, ultralytics=ultralytics.__version__)
    if args.resume:
        with (run / 'previous_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(old) + '\n')
        info.update(resumed=True, previous_elapsed_seconds=float(old.get('elapsed_seconds', 0)),
                    previous_attempt_status=old['status'])
    write_json(metadata, info)
    started = time.monotonic()
    try:
        model = YOLO(str(run / 'weights/last.pt' if args.resume else weights))
        model.train(data=str(run / 'data.yaml'), project=str(runs), name=name, exist_ok=True,
                    seed=recipe['seed'], device='0', resume=args.resume, **protocol)
        verify_dataset(data, manifest)
        info.update(status='complete', best_sha256=file_hash(run / 'weights/best.pt'))
    except BaseException as error:
        info.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        info['attempt_elapsed_seconds'] = time.monotonic() - started
        info['elapsed_seconds'] = info.get('previous_elapsed_seconds', 0) + info['attempt_elapsed_seconds']
        write_json(metadata, info)


if __name__ == '__main__':
    main()
