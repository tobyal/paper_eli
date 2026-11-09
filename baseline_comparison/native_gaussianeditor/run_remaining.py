"""Run the four approved native-flow cases with two isolated GPU queues."""
import concurrent.futures
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'
CODE = BASE / 'native_gaussianeditor'
OUT = BASE / 'gaussianeditor_native'
lock = threading.Lock()
states = {}


def record(case, **values):
    with lock:
        states.setdefault(case, {}).update(values)
        path = OUT / 'remaining_batch_status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_at=time.time(), cases=states), indent=2))
        temporary.replace(path)


def queue(gpu, cases):
    for case in cases:
        record(case, status='waiting_for_memory', gpu=gpu)
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 26000:
                break
            time.sleep(20)
        folder = OUT / case
        log = folder / 'logs/batch_supervisor.log'
        assert not log.exists(), 'Do not overwrite an existing attempt'
        record(case, status='running', started_at=time.time())
        with log.open('w') as stream:
            child = subprocess.Popen(['python3', str(CODE / 'run.py'), case, '--gpu', gpu],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record(case, pid=child.pid)
            rc = child.wait()
        if rc:
            record(case, status='failed', exit_code=rc)
            continue
        placement = folder / 'add/placement.json'
        if placement.exists():
            shutil.copy2(placement, folder / 'records/placement.json')
        with (folder / 'logs/verification.log').open('w') as stream:
            check = subprocess.run([str(ROOT / 'envs/gsle/bin/python'), str(CODE / 'verify.py'), case],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record(case, status='complete' if check.returncode == 0 else 'verification_failed',
               verification_exit_code=check.returncode, finished_at=time.time())


def main():
    for case in ['case02', 'case04', 'case07', 'case10']:
        subprocess.run(['python3', str(CODE / 'prepare.py'), case], cwd=ROOT, check=True)
        record(case, status='queued')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, '2', ['case02', 'case07']),
                   pool.submit(queue, '3', ['case04', 'case10'])]
        for future in futures:
            future.result()
    print(json.dumps(states, indent=2), flush=True)


if __name__ == '__main__':
    main()
