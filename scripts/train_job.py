"""Persistent bounded smoke/full training job; no cloud credentials needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def interrupt(process):
    if os.name == 'nt':
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolo26n', 'yolo26m'], required=True)
    parser.add_argument('--max-hours', type=float, default=6)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    control = ROOT / 'control'
    control.mkdir(exist_ok=True)
    status_file = control / 'status.json'
    if status_file.exists():
        raise SystemExit('Existing job status: refusing to overwrite; inspect before resuming')
    state = {'model': args.model, 'status': 'running', 'started_unix': time.time(), 'pid': os.getpid()}
    write(status_file, state)
    commands = []
    if args.bootstrap:
        commands.append(('bootstrap', [sys.executable, '-u', 'scripts/bootstrap_training.py']))
    command = [sys.executable, '-u', '-m', 'runners.train_d3', '--model', args.model,
               '--weights-root', 'weights', '--device', '0']
    commands += [('smoke', command + ['--smoke']), ('train', command)]
    try:
        for stage, cmd in commands:
            state.update(stage=stage, stage_started_unix=time.time())
            write(status_file, state)
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            with (control / (stage + '.log')).open('w') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **options)
                state['child_pid'] = child.pid
                write(status_file, state)
                while child.poll() is None:
                    if (control / 'stop-request').exists() or time.time() - state['started_unix'] > args.max_hours * 3600:
                        interrupt(child)
                        raise RuntimeError('Job stopped by request or wall-clock limit')
                    time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'{stage} exited {child.returncode}')
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
    finally:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(control / 'environment.txt').open('w'), check=False)
        checksums = {}
        for directory in [ROOT / 'runs/d3', control]:
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or '.cache' in path.parts or path.name in {'status.json', 'results.json', 'heartbeat', 'results.tmp'}:
                    continue
                checksum = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block)
                checksums[path.relative_to(ROOT).as_posix()] = checksum.hexdigest()
        write(control / 'results.json', {'files': checksums, 'model': args.model})
        state['finished_unix'] = time.time()
        write(status_file, state)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
