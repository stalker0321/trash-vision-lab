"""Back up the unattended Windows D3 Nano run and verify its final artifact manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from vast_watch import verify_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['training', 'evaluation'], default='training')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes', 'msi-laptop-1']
    scp = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes']
    source = 'D:/trash-vision-d3/'
    control_dir = 'control' if args.task == 'training' else 'control-evaluation'
    runs_dir = 'runs/d3' if args.task == 'training' else 'runs/evaluation'
    archive_name = 'results-final.tar' if args.task == 'training' else 'evaluation-results-final.tar'
    last_backup = 0
    summary = {'status': 'monitoring'}
    def ps(command, timeout=60):
        result = subprocess.run(ssh + ['powershell -NoProfile -Command "' + command + '"'],
                                capture_output=True, text=True, timeout=timeout, check=True)
        return result.stdout
    while True:
        try:
            status = json.loads(ps(f"Get-Content -Raw -LiteralPath 'D:/trash-vision-d3/{control_dir}/status.json'"))
            summary.update(status='monitoring', job=status)
            terminal = status.get('status') in {'complete', 'failed'}
            if terminal:
                command = f"$ErrorActionPreference='Stop'; Set-Location 'D:/trash-vision-d3'; tar.exe --exclude='.cache' -cf '{archive_name}' {control_dir} {runs_dir}; if ($LASTEXITCODE -ne 0) {{ throw 'Archive failed' }}; (Get-FileHash '{archive_name}' -Algorithm SHA256).Hash"
                expected = ps(command, timeout=300).strip().lower()
                archive = args.output / archive_name
                subprocess.run(scp + ['msi-laptop-1:' + source + archive_name, str(archive)], check=True, timeout=600)
                checksum = hashlib.sha256()
                with archive.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block)
                if checksum.hexdigest() != expected:
                    raise ValueError('Final archive hash mismatch')
                artifacts = args.output / 'artifacts'
                artifacts.mkdir(exist_ok=True)
                with tarfile.open(archive) as bundle:
                    bundle.extractall(artifacts, filter='data')
                count = verify_results(artifacts, f'{control_dir}/results.json', (runs_dir + '/', control_dir + '/'))
                summary.update(status='results_verified', job_outcome=status['status'], verified_files=count,
                               archive_sha256=expected, finished_unix=time.time())
            elif status.get('stage') == 'train' and time.time() - last_backup > 600:
                latest = args.output / 'latest'
                latest.mkdir(exist_ok=True)
                run = source + 'runs/d3/yolo26n_d3_s42_wd001/'
                for name in ['results.csv', 'run.json', 'weights/best.pt', 'weights/last.pt']:
                    destination = latest / name
                    destination.parent.mkdir(exist_ok=True)
                    result = subprocess.run(scp + ['msi-laptop-1:' + run + name, str(destination)],
                                            capture_output=True, text=True, timeout=180)
                    if result.returncode:
                        summary['backup_pending'] = name
                last_backup = time.time()
                summary['last_backup_unix'] = last_backup
            summary['updated_unix'] = time.time()
            temporary = args.output / 'watch-status.tmp'
            temporary.write_text(json.dumps(summary, indent=2) + '\n')
            temporary.replace(args.output / 'watch-status.json')
            if terminal:
                print(json.dumps(summary), flush=True)
                return
        except Exception as error:
            summary.update(last_error=type(error).__name__, updated_unix=time.time())
            (args.output / 'watch-status.json').write_text(json.dumps(summary, indent=2) + '\n')
        time.sleep(30)


if __name__ == '__main__':
    main()
