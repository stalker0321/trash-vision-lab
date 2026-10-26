"""Export Nano320 using private, letterboxed copies of the frozen train calibration split."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from datasets.build_d3 import digest, file_hash, pool_path, validate_splits, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-metadata', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    data = args.data_root.resolve()
    manifest = json.loads((data / 'dataset.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('Dataset manifest fingerprint is invalid')
    recipe = json.loads(Path('configs/d3_training.json').read_text())
    if manifest['fingerprint'] != recipe['dataset_fingerprint']:
        raise ValueError('Use the frozen D3 calibration membership')
    validate_splits(manifest)
    index = {row['id']: row for row in manifest['images']}
    ids = manifest['splits']['calibration']
    if len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError('Expected 256 unique real training calibration images')
    checkpoint_hash = file_hash(args.weights)
    if args.run_metadata:
        run = json.loads(args.run_metadata.read_text())
        if (run['status'] != 'complete' or run['model'] != 'yolo26n' or
                run['dataset_fingerprint'] != manifest['fingerprint'] or run['best_sha256'] != checkpoint_hash):
            raise ValueError('D3 export requires a completed, matching Nano training record')
    elif checkpoint_hash != 'c38dd733b3ed585b99e325f8e57baf7bb1df3edcb544a66e511026b4ae0459e8':
        raise ValueError('Provide --run-metadata for D3; the only standalone checkpoint allowed is historical D2 Nano')
    report = dict(weights_sha256=checkpoint_hash, dataset_fingerprint=manifest['fingerprint'],
                  target='rv1103', imgsz=320, quantize=8, calibration_ids=ids,
                  preprocessing='OpenCV BGR decode, LetterBox320/114, lossless PNG copies; RKNN RGB / 255',
                  coordinate_contract='normalized cxcywh + four sigmoid class scores, 1x8x2100',
                  hardware_validation='pending: compiled artifact is not a board accuracy/performance result')
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite export: {output}')
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the separate Linux Python 3.12 RKNN environment')
    required = {'torch': '2.4.0', 'torchvision': '0.19.0', 'ultralytics': '8.4.143',
                'rknn-toolkit2': '2.3.2', 'numpy': '1.26.4', 'onnx': '1.18.0',
                'onnxruntime': '1.22.1', 'onnxslim': '0.1.96', 'opencv-python': '4.11.0.86'}
    versions = {name: importlib.metadata.version(name) for name in required}
    if any(versions[k].split('+')[0] != v for k, v in required.items()):
        raise RuntimeError('Restore the pinned, separate export environment')
    output.mkdir(parents=True)
    (output / '.cache/ultralytics').mkdir(parents=True)
    (output / '.cache/matplotlib').mkdir(parents=True)
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(output / '.cache/ultralytics'),
                      MPLCONFIGDIR=str(output / '.cache/matplotlib'), OMP_NUM_THREADS='2')
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO, settings
    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.exporter import Exporter
    from ultralytics.utils.export.rknn import onnx2rknn
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    settings.update({'sync': False})
    letterbox = LetterBox((320, 320), auto=False, stride=32, scaleup=True)
    calibration = output / 'calibration'
    calibration.mkdir()
    paths, records = [], []
    for number, image_id in enumerate(ids):
        row = index[image_id]
        source = pool_path(data, row['image'])
        if file_hash(source) != row['image_sha256']:
            raise ValueError(f'Calibration source changed: {image_id}')
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode calibration image: {image_id}')
        destination = calibration / f'{number:04d}.png'
        if not cv2.imwrite(str(destination), letterbox(image=image)):
            raise OSError('Cannot save calibration copy')
        paths.append(destination)
        records.append(dict(id=image_id, source_sha256=row['image_sha256'], png_sha256=file_hash(destination)))
    image_list = calibration / 'images.txt'
    image_list.write_text(''.join(str(p) + '\n' for p in paths))
    # The export consumes this image list directly: no dataset scanner can rewrite source JPEGs.
    copied_weights = output / 'nano.pt'
    shutil.copyfile(args.weights, copied_weights)
    model = YOLO(str(copied_weights))
    if model.task != 'detect' or [model.names[i] for i in range(4)] != manifest['names']:
        raise ValueError('Expected the four-material detection checkpoint')

    class CalibrationExporter(Exporter):
        def export_rknn(self, prefix='RKNN:'):
            self.im = self.im[:1]
            onnx_path = self.export_onnx()
            onnx.checker.check_model(onnx.load(onnx_path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 2
            session = ort.InferenceSession(onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
            # torch.onnx restores the wrapper's training flag after tracing.
            self.model.eval()
            parity = []
            for path in paths[:8]:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
                with torch.inference_mode():
                    prediction = self.model(torch.from_numpy(tensor))
                    if isinstance(prediction, (tuple, list)):
                        prediction = prediction[0]
                    reference = prediction.cpu().numpy().copy()
                reference[:, :4, :] /= 320
                actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                if actual.shape != (1, 8, 2100) or not np.isfinite(actual).all():
                    raise ValueError('Unexpected normalized ONNX output')
                delta = np.abs(actual - reference)
                if not np.allclose(actual, reference, rtol=1e-3, atol=1e-4):
                    raise ValueError(f'Normalized ONNX/PyTorch mismatch: {float(delta.max())}')
                parity.append(dict(image=path.name, maximum_absolute_error=float(delta.max()),
                                   mean_absolute_error=float(delta.mean())))
            report['onnx_parity'] = parity
            report['onnx_sha256'] = file_hash(Path(onnx_path))
            return onnx2rknn(onnx_file=onnx_path, output_dir=output / 'model', name='rv1103',
                             quantize=8, dataset=image_list, metadata=self.metadata, prefix=prefix, batch=1)

    exporter = CalibrationExporter(overrides=dict(format='rknn', name='rv1103', quantize=8,
        imgsz=320, batch=1, device='cpu', opset=19, simplify=True, dynamic=False, nms=False, data=str(image_list)))
    exporter(model=model.model)
    models = list((output / 'model').glob('*.rknn'))
    if len(models) != 1:
        raise ValueError('Expected exactly one RKNN artifact')
    for image_id in ids:
        row = index[image_id]
        if file_hash(pool_path(data, row['image'])) != row['image_sha256']:
            raise ValueError(f'Export modified a source image: {image_id}')
    if file_hash(args.weights) != checkpoint_hash:
        raise ValueError('Export modified the original checkpoint')
    report.update(status='complete', rknn_sha256=file_hash(models[0]), environment=versions,
                  python=platform.python_version(), calibration_copies=records)
    write_json(output / 'export.json', report)
    with (output / 'environment.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    print(json.dumps({'export': str(models[0]), 'sha256': report['rknn_sha256'], 'board_test': 'pending'}))


if __name__ == '__main__':
    main()
