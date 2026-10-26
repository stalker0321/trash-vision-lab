"""Monitor one explicitly recorded rental; verify downloaded results before destruction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def api(path, key, method='GET', body=None):
    request = Request('https://console.vast.ai/api/v0' + path,
                      data=None if body is None else json.dumps(body).encode(), method=method,
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=25) as response:
        return json.load(response)


def verify_results(directory, manifest_path='control/results.json', allowed_prefixes=('runs/d3/', 'control/')):
    manifest = json.loads((directory / manifest_path).read_text())
    if not manifest.get('files'):
        raise ValueError('Empty results manifest')
    for relative, expected in manifest['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not relative.startswith(allowed_prefixes):
            raise ValueError('Result path escapes allowed directories')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
        if checksum.hexdigest() != expected:
            raise ValueError(f'Result checksum differs: {relative}')
    return len(manifest['files'])


def needs_emergency_stop(error, failures, config, now):
    if now >= float(config['deadline_unix']):
        return True
    guarded_transport = (config.get('remote_deadline_guard_confirmed', False) and
                         isinstance(error, (URLError, TimeoutError, subprocess.TimeoutExpired,
                                            subprocess.CalledProcessError)))
    return failures >= 10 and not guarded_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = args.config.resolve().parent
    key_path = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'vastai/vast_api_key'
    key = key_path.read_text().strip()
    instance_id = int(config['instance_id'])
    remote_root = '/workspace/trash-vision-d3'
    artifacts = directory / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    summary = {'instance_id': instance_id, 'status': 'monitoring'}
    last_backup = 0
    stop_requested_at = None
    failures = 0
    def record():
        summary['updated_unix'] = time.time()
        temporary = directory / 'watch-status.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(directory / 'watch-status.json')
    while True:
        try:
            try:
                row = api(f'/instances/{instance_id}/?owner=me', key).get('instances')
            except HTTPError as error:
                if error.code != 404:
                    raise
                row = None
            if row is None:
                summary['status'] = 'destroyed' if summary.get('verified_files') else 'instance_absent'
                summary['finished_unix'] = time.time()
                record()
                print(json.dumps(summary), flush=True)
                return
            if row.get('label') != config['label']:
                raise SystemExit('Instance label mismatch; refusing to manage another rental')
            account = api('/users/current/', key)
            credit = float(account.get('credit', 0))
            summary.update(credit=credit, instance_status=row.get('actual_status'))
            timed_out = time.time() >= float(config['deadline_unix'])
            spent = float(config['starting_credit']) - credit
            limit = spent >= float(config['max_spend_usd']) or credit <= float(config['credit_reserve_usd'])
            if row.get('actual_status') != 'running':
                if time.time() > float(config['created_unix']) + 2700 or timed_out or limit:
                    api(f'/instances/{instance_id}/', key, 'PUT', {'state': 'stopped'})
                    summary['status'] = 'stopped_needs_recovery'
                record()
                time.sleep(30)
                continue
            ports = row.get('ports') or {}
            direct = (ports.get('22/tcp') or [{}])[0].get('HostPort')
            host = row.get('public_ipaddr') if direct else row.get('ssh_host')
            port = str(direct or row.get('ssh_port'))
            if not host or port == 'None':
                raise RuntimeError('SSH address is not ready')
            ssh = ['ssh', '-i', str(Path.home() / '.ssh/id_ed25519'), '-p', port,
                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=3', '-o', 'StrictHostKeyChecking=accept-new',
                   '-o', 'UserKnownHostsFile=' + str(directory / 'known_hosts')]
            target = 'root@' + host
            command = f'mkdir -p {remote_root}/control; touch {remote_root}/control/heartbeat; '
            if timed_out or limit:
                command += f'touch {remote_root}/control/stop-request; '
                stop_requested_at = stop_requested_at or time.time()
            command += (f'if [ -f {remote_root}/control/status.json ]; then cat {remote_root}/control/status.json; '
                        f'elif [ -f {remote_root}/control/staging.json ]; then cat {remote_root}/control/staging.json; '
                        'else echo \'{"status":"preparing"}\'; fi')
            response = subprocess.run(ssh + [target, command], capture_output=True, text=True, timeout=60, check=True)
            status = json.loads(response.stdout)
            status.setdefault('status', 'preparing')
            summary.update(job=status, host=host, port=int(port), status='monitoring')
            if status['status'] == 'preparing' and status.get('stage') == 'failed':
                api(f'/instances/{instance_id}/', key, 'PUT', {'state': 'stopped'})
                summary['status'] = 'stopped_preparation_failed'
                record()
                time.sleep(30)
                continue
            terminal = status.get('status') in {'complete', 'failed'}
            if terminal or (status.get('status') == 'running' and time.time() - last_backup > 600):
                for subdir in ['control', 'runs/d3']:
                    destination = artifacts / subdir
                    destination.mkdir(parents=True, exist_ok=True)
                    copy = subprocess.run(['rsync', '-a', '--partial', '--timeout=90', '--exclude=.cache',
                                           '-e', shlex.join(ssh), f'{target}:{remote_root}/{subdir}/', str(destination) + '/'],
                                          capture_output=True, text=True, timeout=600)
                    if copy.returncode and terminal:
                        raise RuntimeError('Final result transfer failed')
                last_backup = time.time()
                summary['last_backup_unix'] = last_backup
            if terminal:
                count = verify_results(artifacts)
                summary.update(status='results_verified', verified_files=count)
                verified_at = summary.setdefault('results_verified_unix', time.time())
                hold = min(300, max(0, float(config.get('release_transfer_hold_seconds', 0))))
                if status['status'] == 'complete' and hold:
                    acknowledgment = artifacts / 'control/release-transfer-complete.json'
                    run = json.loads((artifacts / 'runs/d3/yolo26m_d3_s42_wd001/run.json').read_text())
                    ack = json.loads(acknowledgment.read_text()) if acknowledgment.exists() else {}
                    received = ack.get('status') == 'ready' and ack.get('medium_sha256') == run['best_sha256']
                    summary['release_transfer_received'] = received
                    if not received and not timed_out and not limit and time.time() - verified_at < hold:
                        record()
                        time.sleep(15)
                        continue
                record()
                result = api(f'/instances/{instance_id}/', key, 'DELETE', {})
                if not result.get('success'):
                    raise RuntimeError('Destroy request was not acknowledged')
                summary.update(status='destroy_requested', job_outcome=status['status'])
                record()
                time.sleep(5)
                continue
            if stop_requested_at and time.time() - stop_requested_at > 180:
                api(f'/instances/{instance_id}/', key, 'PUT', {'state': 'stopped'})
                summary['status'] = 'stopped_needs_recovery'
            failures = 0
            record()
        except Exception as error:
            failures += 1
            summary.update(last_error=type(error).__name__, consecutive_errors=failures)
            record()
            if needs_emergency_stop(error, failures, config, time.time()):
                try:
                    row = api(f'/instances/{instance_id}/?owner=me', key).get('instances')
                    if row and row.get('label') == config['label']:
                        api(f'/instances/{instance_id}/', key, 'PUT', {'state': 'stopped'})
                        summary['status'] = 'stopped_needs_recovery'
                        record()
                except Exception:
                    pass
        time.sleep(30)


if __name__ == '__main__':
    main()
