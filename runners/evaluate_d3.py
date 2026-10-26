"""Evaluate a fixed four-material checkpoint on the frozen reviewed holdout."""
import argparse
import json
import os
from pathlib import Path
import platform

from datasets.build_d3 import digest, file_hash, pool_path, validate_splits, write_json
from runners.train_d3 import readonly_image_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, choices=[320, 640], required=True)
    parser.add_argument('--split', choices=['val', 'test', 'hard_test'], default='test')
    parser.add_argument('--data-root', type=Path, default=Path('data/versions/d3'))
    parser.add_argument('--recipe', type=Path, default=Path('configs/d3_training.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    data = args.data_root.resolve()
    manifest = json.loads((data / 'dataset.json').read_text())
    recipe = json.loads(args.recipe.read_text())
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('Invalid dataset fingerprint')
    if manifest['fingerprint'] != recipe['dataset_fingerprint']:
        raise ValueError('Evaluation dataset differs from the frozen D3')
    validate_splits(manifest)
    index = {r['id']: r for r in manifest['images']}
    rows = [index[i] for i in manifest['splits'][args.split]]
    protocol = {'imgsz': args.imgsz, 'batch': 8, 'workers': 0, 'conf': .001,
                'iou': .7, 'max_det': 300, 'half': False, 'augment': False}
    summary = {'weights_name': args.weights.name, 'weights_sha256': file_hash(args.weights),
               'dataset_fingerprint': manifest['fingerprint'], 'split': args.split,
               'images': len(rows), 'boxes': sum(r['bbox_count'] for r in rows),
               'protocol': protocol, 'class_names': manifest['names']}
    if args.plan_only:
        print(json.dumps(summary, indent=2))
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    if not args.device.isdigit() or '.'.join(platform.python_version_tuple()[:2]) != recipe['environment']['python']:
        raise RuntimeError('Use Python 3.12 and one CUDA device ID')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    for key, name in [('YOLO_CONFIG_DIR', 'ultralytics'), ('MPLCONFIGDIR', 'matplotlib')]:
        os.environ[key] = str(output.parent / '.cache' / name)
    import torch
    import ultralytics
    import ultralytics.data.utils
    from ultralytics import YOLO, settings
    if torch.__version__.split('+')[0] != recipe['environment']['torch'] or ultralytics.__version__ != recipe['environment']['ultralytics']:
        raise RuntimeError('Use the pinned training environment')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    def verify():
        for row in rows:
            for kind in ['image', 'label']:
                if file_hash(pool_path(data, row[kind])) != row[kind + '_sha256']:
                    raise ValueError(f'Evaluation data changed: {row["id"]} / {kind}')
    verify()
    inputs = output.parent / '.inputs' / output.name
    inputs.mkdir(parents=True, exist_ok=False)
    image_list = inputs / 'images.txt'
    image_list.write_text(''.join(str(data / r['image']) + '\n' for r in rows))
    yaml = inputs / 'data.yaml'
    write_json(yaml, {'path': str(data), 'train': str(image_list), 'val': str(image_list), 'names': manifest['names']})
    ultralytics.data.utils.check_image = readonly_image_check
    settings.update({'sync': False})
    model = YOLO(str(args.weights.resolve()))
    if [model.names[i] for i in range(len(model.names))] != manifest['names']:
        raise ValueError('Checkpoint class names/order do not match the four-material task')
    metrics = model.val(data=str(yaml), split='val', device='0', project=str(output.parent),
                        name=output.name, exist_ok=False, plots=True, save_json=True, **protocol)
    verify()
    per_class = {name: None for name in manifest['names']}
    for position, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(position)
        per_class[manifest['names'][int(class_id)]] = {'precision': float(precision), 'recall': float(recall),
                                                      'ap50': float(ap50), 'ap50_95': float(ap)}
    summary.update(metrics={k: float(v) for k, v in metrics.results_dict.items()}, per_class=per_class,
                   speed_ms=metrics.speed, gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                   torch=torch.__version__, ultralytics=ultralytics.__version__, platform=platform.platform(),
                   note='Speed is validator timing, not a controlled deployment benchmark. Hard test is a subset of test.')
    write_json(Path(metrics.save_dir) / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
