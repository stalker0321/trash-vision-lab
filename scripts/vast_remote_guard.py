"""Instance-scoped fallback: stop compute on lost controller/deadline; retain data."""
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    instance = int(os.environ['CONTAINER_ID'])
    token = os.environ['CONTAINER_API_KEY']
    deadline = float(os.environ['TVL_DEADLINE'])
    control = Path('/workspace/trash-vision-d3/control')
    control.mkdir(parents=True, exist_ok=True)
    (control / 'remote-guard.json').write_text(json.dumps({'pid': os.getpid(), 'instance_id': instance, 'deadline_unix': deadline}) + '\n')
    started = time.time()
    while True:
        heartbeat = control / 'heartbeat'
        last_seen = heartbeat.stat().st_mtime if heartbeat.exists() else started
        if time.time() > deadline or time.time() - last_seen > 1200:
            (control / 'stop-request').touch()
            print('Fallback stop requested: deadline or controller heartbeat timeout', flush=True)
            time.sleep(90)
            request = Request(f'https://console.vast.ai/api/v0/instances/{instance}/',
                              data=json.dumps({'state': 'stopped'}).encode(), method='PUT',
                              headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            for _ in range(12):
                try:
                    with urlopen(request, timeout=20) as response:
                        result = json.load(response)
                    if result.get('success'):
                        return
                except Exception as error:
                    print('Stop request retry:', type(error).__name__, flush=True)
                time.sleep(10)
        time.sleep(30)


if __name__ == '__main__':
    main()
