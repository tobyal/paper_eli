"""Continue the two coarse-only jobs after their saved 6500-step completion."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time,json,subprocess
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
def finish(case,gpu):
 p=BASE/'gsle'/case/'records/status.json'
 while True:
  data=json.loads(p.read_text())
  if data['stages'].get('coarse',{}).get('status')=='complete':break
  if data['status']=='failed':print('Requires recovery',case,flush=True);return
  time.sleep(15)
 print('CONTINUE',case,'GPU',gpu,flush=True)
 rc=subprocess.run(['python3',str(BASE/'scripts/run_gsle.py'),case,'--gpu',gpu],cwd=ROOT).returncode
 print('FINISHED',case,rc,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(finish,'case02','0'),pool.submit(finish,'case10','1')]
 for f in futures:f.result()
