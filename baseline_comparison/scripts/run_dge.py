"""Official DGE local editing, with exact per-case A/B and isolated caches."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',required=True);a=p.parse_args()
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';out=BASE/'dge'/a.case
task=json.loads((out/'task.json').read_text());models=json.loads((ROOT/'records/core_model_downloads.json').read_text())
env=os.environ.copy();env.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'scripts'),TARGET_GEOMETRY=task['instance_geometry'],INSTANCE_SELECTION_LOG=str(out/'records/instance_selection.jsonl'))
command=[str(ROOT/'envs/dge/bin/python'),'launch.py','--config','configs/dge.yaml','--train','--gpu','0',
 'data.source='+task['native_view_source'],'system.gs_source='+str(ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'),
 'system.seg_prompt='+task['A'],'system.prompt_processor.prompt='+task['dge_prompt'],
 'exp_root_dir='+str(out/'official'),'tag='+a.case,'use_timestamp=false','seed=42',
 'system.cache_dir='+str(out/'edit_cache'),'system.cache_overwrite=true','system.local_edit=false',
 'system.mask_thres=0.5','system.loggers.wandb.enable=false',
 'system.guidance.ip2p_name_or_path='+models['timbrooks/instruct-pix2pix']['path'],
 'system.guidance.ddim_scheduler_name_or_path='+models['CompVis/stable-diffusion-v1-4']['path'],
 'system.prompt_processor.pretrained_model_name_or_path='+models['stable-diffusion-v1-5/stable-diffusion-v1-5']['path']]
status=out/'records/status.json';assert not status.exists(),'Existing attempt is immutable; archive before retrying'
record={'status':'running','method':'dge','case':a.case,'argv':command,'cwd':str(ROOT/'repos/DGE'),'started_at':time.time(),'gpu':a.gpu,'target':task['A'],'replacement':task['B'],'local_edit':False,'local_editing_mechanism':'Official README: seg_prompt semantic tracing and Gaussian gradient masks; render complete scene','mask_threshold':.5,'iterations':1000}
status.write_text(json.dumps(record,indent=2))
with (out/'logs/train.log').open('w') as log:
 process=subprocess.Popen(command,cwd=ROOT/'repos/DGE',env=env,stdout=log,stderr=subprocess.STDOUT)
 record['pid']=process.pid;status.write_text(json.dumps(record,indent=2));code=process.wait()
record.update(exit_code=code,seconds=time.time()-record['started_at'],status='training_complete' if code==0 else 'failed')
if code==0:
    files=list((out/'official').rglob('last.ply'));assert len(files)==1
    record['scene_ply']=str(files[0]);(out/'scene_final.ply').symlink_to(files[0])
status.write_text(json.dumps(record,indent=2));print(record['status'],a.case,flush=True);raise SystemExit(code)
