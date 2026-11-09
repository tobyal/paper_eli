"""Run all existing GSLE stages on a new case; save per-stage real status."""
import argparse,os,json,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',required=True);p.add_argument('--through',default='refine');a=p.parse_args()
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';out=BASE/'gsle'/a.case;code=BASE/'vendor/gsle'
env=os.environ.copy();env.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,GSLE_OUTPUT=str(out),GSLE_RECORDS=str(out/'records'),GSLE_CONFIG=str(out/'config.json'),
 HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1')
stages=[('reference','gsle_langsplat','reference_proposals.py',[]),('query','gsle_langsplat','query_language.py',[]),
 ('localization','gsle','localize.py',[]),('region','gsle','prepare_edit_region.py',[]),('avp','gsle','avp.py',[]),
 ('inpainting','gsle','inpaint.py',[]),('ranking','gsle_langsplat','rank_inpainting.py',[]),('foreground','gsle_langsplat','sam_foreground.py',[]),
 ('coarse','gsle','train_object.py',['coarse','--microbatch','2']),('composition','gsle','compose.py',['coarse']),
 ('refine','gsle','train_object.py',['refine','--microbatch','4']),('final_composition','gsle','compose.py',['refine'])]
status=out/'records/status.json';record=json.loads(status.read_text()) if status.exists() else {'method':'gsle','case':a.case,'stages':{},'started_at':time.time(),'gpu':a.gpu}
record['gpu']=a.gpu
if a.through!='refine':stages=stages[:next(i for i,v in enumerate(stages) if v[0]==a.through)+1]
for name,venv,script,extra in stages:
 if record['stages'].get(name,{}).get('status')=='complete':continue
 if name=='refine':
  while True:
   free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free>=22000:break
   record.update(status='waiting_for_shared_gpu_memory',active_stage=name,free_memory_MiB=free);status.write_text(json.dumps(record,indent=2));time.sleep(15)
 command=[str(ROOT/'envs'/venv/'bin/python'),str(code/script)]+extra
 record.update(status='running',active_stage=name);record['stages'][name]={'status':'running','argv':command,'started_at':time.time()};status.write_text(json.dumps(record,indent=2))
 with (out/'logs'/(name+'.log')).open('w') as log:
  process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);record['pid']=process.pid;status.write_text(json.dumps(record,indent=2));rc=process.wait()
 row=record['stages'][name];row.update(exit_code=rc,seconds=time.time()-row['started_at'],status='complete' if rc==0 else 'failed');record['status']='running' if rc==0 else 'failed';status.write_text(json.dumps(record,indent=2))
 if rc:print(a.case,name,'FAILED',flush=True);raise SystemExit(rc)
 print(a.case,name,'COMPLETE',flush=True)
if a.through!='refine':
 record.update(status='prepared_through_'+a.through);status.write_text(json.dumps(record,indent=2));raise SystemExit(0)
final=out/'final/scene.ply';assert final.is_file(),list((out/'final').glob('*'))
if not (out/'scene_final.ply').exists():(out/'scene_final.ply').symlink_to(final)
record.update(status='training_complete',scene_ply=str(final),seconds=time.time()-record['started_at']);status.write_text(json.dumps(record,indent=2))
