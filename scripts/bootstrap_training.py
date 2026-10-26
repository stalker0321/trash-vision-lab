"""Restore the pinned training environment in the selected dedicated Python."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Python 3.12 is required')
    check = subprocess.run([sys.executable, '-c',
        'import torch,torchvision; assert torch.__version__.split("+")[0]=="2.8.0"; '
        'assert torchvision.__version__.split("+")[0]=="0.23.0"; assert torch.cuda.is_available()'])
    if check.returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                        'torch==2.8.0', 'torchvision==0.23.0', '--index-url',
                        'https://download.pytorch.org/whl/cu128'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    '-r', str(ROOT / 'configs/requirements-training-linux.lock'),
                    'torch==2.8.0', 'torchvision==0.23.0'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    import torch, torchvision, ultralytics
    assert torch.__version__.split('+')[0] == '2.8.0'
    assert torchvision.__version__.split('+')[0] == '0.23.0'
    assert ultralytics.__version__ == '8.4.143'
    assert torch.cuda.is_available()
    print('Ready:', sys.version, torch.__version__, torchvision.__version__, ultralytics.__version__, torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()
