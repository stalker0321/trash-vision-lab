"""Run the two fixed Nano ablations sequentially on a standalone Windows worker."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from datasets.build_d3 import file_hash


class StageFailed(RuntimeError):
    pass


class QueueStopped(RuntimeError):
    pass


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        if os.name == 'nt':
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def execute_experiments(jobs, invoke):
    """Preflight both before full training; an individual failure doesn't block the next."""
    for job in jobs:
        try:
            invoke(job, 'smoke')
            job['status'] = 'ready'
        except StageFailed as error:
            job.update(status='failed', error=str(error))
    for job in jobs:
        if job['status'] == 'failed':
            continue
        try:
            job['status'] = 'training'
            invoke(job, 'train')
            job['training_complete'] = True
            job['status'] = 'validating'
            invoke(job, 'val')
            job['status'] = 'complete'
        except StageFailed as error:
            job.update(status='failed', error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/overnight_queue.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    base = args.base_root.resolve()
    config = json.loads(args.config.read_text())
    control = root / 'control-night'
    control.mkdir(exist_ok=True)
    status_path = control / 'status.json'
    if status_path.exists():
        raise FileExistsError('Queue status already exists; refusing to launch twice')
    parent = base / 'data/versions/d3'
    if json.loads((parent / 'dataset.json').read_text())['fingerprint'] != config['parent_fingerprint']:
        raise ValueError('Unexpected parent dataset')
    variant = root / 'data/versions/d3-drop-lt8'
    state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
             'jobs': [dict(j, status='pending') for j in config['jobs']]}
    atomic_json(status_path, state)
    atomic_json(control / 'queue.json', config)
    deadline = state['started_unix'] + config['total_max_hours']*3600
    power_set = False
    if os.name == 'nt':
        power_set = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not power_set:
            state.update(status='failed', error='Unable to inhibit idle sleep')
            atomic_json(status_path, state)
            raise RuntimeError(state['error'])

    def update():
        state['updated_unix'] = time.time()
        atomic_json(status_path, state)

    def run(command, tag, hours, progress=None):
        if (control / 'stop-request').exists() or time.time() >= deadline:
            raise QueueStopped('Stop request or global wall-clock deadline')
        started = time.time()
        state.update(current=tag, stage_started_unix=started, command=command)
        state.pop('last_epoch', None)
        update()
        options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        with (control / f'{tag}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=root, **options)
            state['child_pid'] = process.pid
            update()
            try:
                while process.poll() is None:
                    if (control / 'stop-request').exists() or time.time() >= deadline:
                        raise QueueStopped('Stop request or global wall-clock deadline')
                    if time.time() - started > hours*3600:
                        raise StageFailed(f'{tag}: stage time limit exceeded')
                    if progress and progress.exists():
                        try:
                            rows = list(csv.DictReader(progress.read_text().splitlines()))
                            if rows:
                                last = {k.strip(): v.strip() for k, v in rows[-1].items() if v is not None}
                                state['last_epoch'] = int(float(last['epoch']))
                                state['last_metrics'] = last
                        except (OSError, ValueError, KeyError):
                            pass
                    update()
                    time.sleep(5)
            except BaseException:
                stop_child(process)
                raise
            finally:
                state.pop('child_pid', None)
                update()
            if process.returncode:
                raise StageFailed(f'{tag}: process exited {process.returncode}; see its log')

    def invoke(job, stage):
        recipe = json.loads((root / job['recipe']).read_text())
        dataset = parent if job['dataset'] == 'parent' else variant
        run_path = root / 'runs/night' / recipe['models']['yolo26n']['run_name']
        if stage in {'smoke', 'train'}:
            command = [sys.executable, '-u', '-m', 'runners.train_d3', '--model', 'yolo26n',
                '--data-root', str(dataset), '--recipe', job['recipe'], '--weights-root', str(base / 'weights'),
                '--runs-root', str(root / 'runs/night'), '--device', '0']
            if stage == 'smoke':
                command.append('--smoke')
            run(command, job['name'] + '-' + stage, 1 if stage == 'smoke' else job['max_train_hours'],
                run_path / 'results.csv' if stage == 'train' else None)
            metadata_path = run_path.with_name(run_path.name + '_smoke') / 'run.json' if stage == 'smoke' else run_path / 'run.json'
            try:
                metadata = json.loads(metadata_path.read_text())
                checkpoint = metadata_path.parent / 'weights/best.pt'
                if metadata['status'] != 'complete' or metadata['dataset_fingerprint'] != recipe['dataset_fingerprint']:
                    raise ValueError('No matching completed training record')
                if file_hash(checkpoint) != metadata['best_sha256']:
                    raise ValueError('Best checkpoint checksum mismatch')
            except (OSError, ValueError, KeyError) as error:
                raise StageFailed(f'{job["name"]}: invalid completed checkpoint: {error}') from error
            if stage == 'train':
                job['best_sha256'] = metadata['best_sha256']
                job['elapsed_seconds'] = metadata['elapsed_seconds']
        else:
            # Always evaluate against the original, complete D3 holdout.
            command = [sys.executable, '-u', '-m', 'runners.evaluate_d3', '--weights', str(run_path / 'weights/best.pt'),
                       '--imgsz', str(recipe['models']['yolo26n']['imgsz']), '--split', 'val',
                       '--data-root', str(parent), '--recipe', 'configs/d3_training.json',
                       '--output', str(root / 'runs/evaluation' / (job['name'] + '_val')), '--device', '0']
            run(command, job['name'] + '-val', .5)
        update()

    try:
        run([sys.executable, '-u', '-m', 'datasets.build_small_box_variant', '--source', str(parent),
             '--manifest', 'manifests/d3-drop-lt8.json', '--output', str(variant),
             '--report', str(control / 'label-deletion.json')], 'prepare-variant', 1)
        execute_experiments(state['jobs'], invoke)
        state['status'] = 'complete' if all(j['status'] == 'complete' for j in state['jobs']) else 'complete_with_errors'
    except BaseException as error:
        state.update(status='stopped' if isinstance(error, QueueStopped) else 'failed', error=repr(error))
    finally:
        state.update(finished_unix=time.time(), current='finalizing')
        update()
        try:
            files = {}
            for directory in [root / 'runs', root / 'configs', root / 'manifests', control]:
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and '.cache' not in path.parts and path.name not in {'status.json', 'results.json'} and path.suffix != '.tmp':
                        files[path.relative_to(root).as_posix()] = file_hash(path)
            atomic_json(control / 'results.json', {'files': files, 'queue_status': state['status']})
            archive = root / 'night-results.tar'
            temporary = archive.with_suffix('.partial')
            with tarfile.open(temporary, 'w') as bundle:
                for relative in files:
                    bundle.add(root / relative, arcname=relative, recursive=False)
                bundle.add(control / 'results.json', arcname='control-night/results.json', recursive=False)
            temporary.replace(archive)
            state['archive_sha256'] = file_hash(archive)
            state['archive_bytes'] = archive.stat().st_size
            archive.with_suffix('.tar.sha256').write_text(state['archive_sha256'] + '  ' + archive.name + '\n')
        except Exception as error:
            state['archive_error'] = repr(error)
        finally:
            state['current'] = 'finished'
            update()
            if power_set:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
