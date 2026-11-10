"""Render the four remaining cases after the inspected case00 completes."""
import json,os,subprocess,time
from pathlib import Path

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
OUT=BASE/'matched_views';CODE=BASE/'matched_rendering';PYTHON=ROOT/'envs/gsle/bin/python'
assert json.loads((OUT/'case00/matching_report.json').read_text())['input_files_unchanged']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONUNBUFFERED='1')
status=dict(started_at=time.time(),training_steps=0,cases={'case00':'complete'})
for case in ['case02','case04','case07','case10']:
    status.update(active_case=case);status['cases'][case]='rendering'
    (OUT/'batch_status.json').write_text(json.dumps(status,indent=2))
    with (OUT/f'{case}_render.log').open('x') as f:
        rc=subprocess.call([str(PYTHON),str(CODE/'render.py'),'--case',case],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    status['cases'][case]='complete' if rc==0 else 'failed'
    (OUT/'batch_status.json').write_text(json.dumps(status,indent=2))
    print(case,status['cases'][case],flush=True)
    if rc:raise SystemExit(rc)
subprocess.run([str(PYTHON),str(CODE/'report.py')],cwd=ROOT,env=env,check=True)
status.update(active_case=None,completed_at=time.time())
(OUT/'batch_status.json').write_text(json.dumps(status,indent=2))
print('All matched views complete',flush=True)
