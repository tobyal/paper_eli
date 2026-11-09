"""Run upstream Delete and Add as separate native stages, retaining checkpoints."""
import argparse,os,json,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',required=True);p.add_argument('--through',default='add');a=p.parse_args()
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';out=BASE/'gaussianeditor'/a.case
task=json.loads((out/'task.json').read_text())
env=os.environ.copy();env.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'scripts'),TARGET_GEOMETRY=task['instance_geometry'],INSTANCE_SELECTION_LOG=str(out/'records/instance_selection.jsonl'),GE_ADD_CACHE=str(out/'add'),PYOPENGL_PLATFORM='egl',EGL_DEVICE_ID=a.gpu,U2NET_HOME=str(ROOT/'models/rembg'),HTTP_PROXY='http://127.0.0.1:17891',HTTPS_PROXY='http://127.0.0.1:17891')
status=out/'records/status.json';record=json.loads(status.read_text()) if status.exists() else {'method':'gaussianeditor','case':a.case,'stages':{},'started_at':time.time(),'gpu':a.gpu}
for name in ['delete','add']:
 if record['stages'].get(name,{}).get('status')=='complete':continue
 # Resource scheduling only: do not change the native model or training settings.
 while True:
  free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free>=22000:break
  record.update(status='waiting_for_shared_gpu_memory',active_stage=name,free_memory_MiB=free);status.write_text(json.dumps(record,indent=2));time.sleep(15)
 cmd=[str(BASE/'envs/gaussianeditor_add/bin/python'),str(BASE/'scripts/ge_native_stage.py'),a.case,name]
 record.update(status='running',active_stage=name);record['stages'][name]={'status':'running','argv':cmd,'started_at':time.time()};status.write_text(json.dumps(record,indent=2))
 logpath=out/'logs'/(name+'.log')
 if logpath.exists():logpath.rename(logpath.with_name(name+f'.previous_{int(time.time())}.log'))
 with logpath.open('w') as log:
  proc=subprocess.Popen(cmd,cwd=BASE/'vendor/GaussianEditor',env=env,stdout=log,stderr=subprocess.STDOUT)
  record['pid']=proc.pid;status.write_text(json.dumps(record,indent=2));rc=proc.wait()
 row=record['stages'][name];row.update(status='complete' if rc==0 else 'failed',exit_code=rc,seconds=time.time()-row['started_at']);record['status']='running' if rc==0 else 'failed';status.write_text(json.dumps(record,indent=2))
 if rc:raise SystemExit(rc)
 if name==a.through and name!='add':raise SystemExit(0)
assert (out/'scene_final.ply').is_file()
record.update(status='training_complete',scene_ply=str(out/'scene_final.ply'),seconds=time.time()-record['started_at']);status.write_text(json.dumps(record,indent=2))
