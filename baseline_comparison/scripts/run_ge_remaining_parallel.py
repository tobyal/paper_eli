"""Schedule remaining native GE cases on two already inspected GPU lanes."""
import json,time,subprocess,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
pending=queue.Queue()
for case in ['case04','case07','case10']:pending.put(case)
def worker(gpu,active_case):
 p=BASE/'gaussianeditor'/active_case/'records/status.json'
 while json.loads(p.read_text())['status']=='running':time.sleep(15)
 while True:
  try:case=pending.get_nowait()
  except queue.Empty:return
  print('START',case,'GPU',gpu,flush=True)
  rc=subprocess.run(['python3',str(BASE/'scripts/run_ge.py'),case,'--gpu',gpu],cwd=ROOT).returncode
  print('FINISHED',case,rc,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(worker,'0','case00'),pool.submit(worker,'1','case02')]
 for f in futures:f.result()
