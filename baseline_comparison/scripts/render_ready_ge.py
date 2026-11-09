"""Use the gap before GSLE case02 refinement for final-camera GE/DGE comparisons."""
import json,time,subprocess,shutil,os
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison'
cases=['case00','case02','case04','case07','case10']
while not all(json.loads((BASE/'gaussianeditor'/c/'records/status.json').read_text())['status']=='training_complete' for c in cases):time.sleep(20)
state=json.loads((BASE/'gsle/case02/records/status.json').read_text())
free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if state.get('active_stage') not in ['coarse','composition'] or free<34000:
 print('No idle render window; defer to finalization',flush=True);raise SystemExit(0)
archive=OUT/'preliminary_v1';archive.mkdir(exist_ok=True)
for case in cases:
 if not (archive/case).exists():shutil.move(str(OUT/case),str(archive/case))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
subprocess.run([str(ROOT/'envs/gsle/bin/python'),str(BASE/'scripts/render_comparison.py')],cwd=ROOT,env=env,check=True)
for name in ['build_report.py','snapshot_reproduction.py','export_offline.py']:
 subprocess.run(['python3',str(BASE/'scripts'/name)],cwd=ROOT,check=True)
print('GE and DGE final-camera comparisons exported',flush=True)
