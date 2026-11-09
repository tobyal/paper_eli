"""After all actual runs finish, archive preliminary views and export common results."""
import json,subprocess,time,hashlib,shutil,os
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison'
spec=json.loads((BASE/'prompts.json').read_text());status_file=BASE/'records/finalization_status.json'
def save(**kwargs):status_file.write_text(json.dumps(kwargs,indent=2))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
while True:
 states={}
 for task in spec['cases']:
  for method in spec['methods']:
   p=BASE/method/task['case'];f=p/'records/status.json'
   d=json.loads(f.read_text()) if f.exists() else {'status':'not_started'}
   states[method+'/'+task['case']]=d['status']
 ready=all(s=='training_complete' for s in states.values())
 save(status='waiting_for_training',states=states,updated_at=time.time())
 if ready:break
 time.sleep(30)
save(status='verifying_preserved_inputs',updated_at=time.time())
checks={'source_scene':{'path':spec['source_scene'],'expected_sha256':spec['source_scene_sha256'],'actual_sha256':sha(spec['source_scene'])},'ours':{},'original_gsle':{}}
assert checks['source_scene']['actual_sha256']==checks['source_scene']['expected_sha256']
for task in spec['cases']:
 case=task['case'];f=OUT/'preliminary_v1'/case/'ours/artifact.json'
 if not f.exists():f=OUT/case/'ours/artifact.json'
 old=json.loads(f.read_text());actual=sha(task['ours_scene']);assert actual==old['sha256'],case
 checks['ours'][case]={'path':task['ours_scene'],'sha256':actual,'unchanged':True}
for name,record in json.loads((BASE/'records/gsle_snapshot.json').read_text()).items():
 actual=sha(ROOT/'gsle'/name);assert actual==record['source'],name
 checks['original_gsle'][name]={'sha256':actual,'unchanged':True}
(BASE/'records/input_preservation_check.json').write_text(json.dumps(checks,indent=2))
archive=OUT/'preliminary_v1';archive.mkdir(exist_ok=True)
for task in spec['cases']:
 case=task['case']
 if not (archive/case).exists():shutil.move(str(OUT/case),str(archive/case))
while True:
 free=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if free>=36000:break
 save(status='waiting_for_render_memory',free_memory_MiB=free,updated_at=time.time());time.sleep(30)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
save(status='rendering_final_common_cameras',updated_at=time.time())
with (BASE/'logs/final_comparison_renders.log').open('w') as log:
 rc=subprocess.call([str(ROOT/'envs/gsle/bin/python'),str(BASE/'scripts/render_comparison.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
if rc:save(status='render_failed',exit_code=rc,updated_at=time.time());raise SystemExit(rc)
for name in ['build_report.py','snapshot_reproduction.py','export_offline.py']:
 subprocess.run(['python3',str(BASE/'scripts'/name)],cwd=ROOT,check=True)
save(status='common_renders_and_export_complete_pending_visual_review',updated_at=time.time())
