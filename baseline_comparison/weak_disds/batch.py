"""Two resource-conscious GPU workers, gated by the reviewed case04 pilot."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(os.environ.get('ROOT','/mnt/sda/ubuntu/lkx/papers/paper_eli'))
BASE=ROOT/'baseline_comparison'
OUT=BASE/'gsle_weak_disds'
PILOT=OUT/'pilot_case04_500'
assert json.loads((PILOT/'records/gate.json').read_text())['passed']
queue=Queue()
for case in ['case02','case07','case10']:queue.put(case)

def worker(gpu,first):
    case=first
    while True:
        while True:
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=18000:break
            print(f'GPU {gpu}: waiting for memory ({free} MiB free)',flush=True)
            time.sleep(30)
        cmd=[str(ROOT/'envs/gsle/bin/python'),str(BASE/'weak_disds/run.py'),case,'--gpu',gpu]
        if case=='case04':cmd+=['--resume-pilot',str(PILOT)]
        print('Starting',case,'GPU',gpu,flush=True)
        with (OUT/f'{case}_launcher.log').open('x') as log:
            rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print('Finished',case,'exit',rc,flush=True)
        if rc:raise RuntimeError((case,rc))
        try:case=queue.get_nowait()
        except Empty:return

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(worker,'3','case04'),pool.submit(worker,'0','case00')]
    for f in futures:f.result()
print('All five runs completed',flush=True)
