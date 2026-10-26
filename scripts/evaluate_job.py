"""Evaluate an explicit checkpoint queue after training, one GPU job at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--wait-for-training', type=Path, default=Path('control/status.json'))
    parser.add_argument('--max-wait-hours', type=float, default=12)
    args = parser.parse_args()
    os.chdir(ROOT)
    queue = json.loads(args.queue.read_text())
    names = [job['name'] for job in queue['jobs']]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[a-z0-9_-]+', n) for n in names):
        raise ValueError('Queue names must be unique safe directory names')
    control = ROOT / 'control-evaluation'
    control.mkdir(exist_ok=True)
    status = control / 'status.json'
    if status.exists():
        raise FileExistsError('Existing evaluation job; inspect it before another launch')
    deadline = time.time() + args.max_wait_hours * 3600
    state = {'status': 'running', 'stage': 'waiting_for_training', 'completed': [], 'started_unix': time.time()}
    write(status, state)
    write(control / 'queue.json', queue)
    (ROOT / 'runs/evaluation').mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def wait():
        if time.time() >= deadline:
            raise TimeoutError('Evaluation queue wait deadline exceeded')
        time.sleep(15)

    try:
        while True:
            training = json.loads(args.wait_for_training.read_text()) if args.wait_for_training.exists() else {}
            if training.get('status') == 'complete':
                break
            if training.get('status') == 'failed':
                raise RuntimeError('Training failed; refusing to evaluate a partial D3 run')
            wait()
        for job in queue['jobs']:
            state.update(stage='waiting_for_weights', current=job['name'])
            write(status, state)
            weights = ROOT / job['weights']
            metadata = ROOT / job['run_metadata'] if job.get('run_metadata') else None
            while not weights.exists() or (metadata and not metadata.exists()):
                wait()
            expected = job.get('sha256')
            if metadata:
                run = json.loads(metadata.read_text())
                if run['status'] != 'complete' or run['dataset_fingerprint'] != queue['dataset_fingerprint']:
                    raise ValueError('D3 checkpoint lacks a matching completed training record')
                expected = run['best_sha256']
            if not expected or checksum(weights) != expected:
                raise ValueError(f'Checkpoint hash mismatch: {job["name"]}')
            for split in ['val', 'test', 'hard_test']:
                name = job['name'] + '_' + split
                state.update(stage='evaluating', current=name)
                write(status, state)
                command = [sys.executable, '-u', '-m', 'runners.evaluate_d3', '--weights', str(weights),
                           '--imgsz', str(job['imgsz']), '--split', split,
                           '--output', str(ROOT / 'runs/evaluation' / name), '--device', '0']
                with (control / (name + '.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                state['completed'].append(name)
                write(status, state)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
    finally:
        files = {}
        for directory in [ROOT / 'runs/evaluation', control]:
            for path in sorted(directory.rglob('*')):
                if path.is_file() and '.cache' not in path.parts and path.name not in {'status.json', 'results.json', 'results.tmp'}:
                    files[path.relative_to(ROOT).as_posix()] = checksum(path)
        write(control / 'results.json', {'files': files})
        state['finished_unix'] = time.time()
        write(status, state)
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
