"""Run official launch.py/configs/dge.yaml with full original COLMAP data."""
import argparse,json,os,subprocess,time
from pathlib import Path
from prepare import prepare,ROOT,BASE,CODE


def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',required=True);a=p.parse_args()
    prepare(a.case);out=BASE/'dge_native'/a.case
    cfg=json.loads((out/'config.json').read_text());prompt=json.loads((out/'prompts.json').read_text())
    models=json.loads((ROOT/'records/core_model_downloads.json').read_text())
    env=os.environ.copy()
    for key in ['TARGET_GEOMETRY','INSTANCE_SELECTION_LOG','GE_TARGET_CAMERA_DEPTH','GE_REUSE_ADD_ASSETS']:env.pop(key,None)
    env.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,DGE_NATIVE_OUTPUT=str(out),HF_HOME=str(ROOT/'models/huggingface'),
               TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
               OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',
               PYTHONPATH=str(BASE/'native_dge')+':'+str(ROOT/'scripts'))
    command=[str(ROOT/'envs/dge/bin/python'),'launch.py','--config','configs/dge.yaml','--train','--gpu','0',
             'data.source='+cfg['data_source'],'system.gs_source='+cfg['source_scene'],
             'system.seg_prompt='+prompt['A'],'system.prompt_processor.prompt='+prompt['edit_prompt'],
             'exp_root_dir='+str(out/'official'),'tag='+a.case,'use_timestamp=false',
             'system.cache_dir='+str(out/'edit_cache'),'system.cache_overwrite=true',
             'system.guidance.ip2p_name_or_path='+models['timbrooks/instruct-pix2pix']['path'],
             'system.guidance.ddim_scheduler_name_or_path='+models['CompVis/stable-diffusion-v1-4']['path'],
             'system.prompt_processor.pretrained_model_name_or_path='+models['stable-diffusion-v1-5/stable-diffusion-v1-5']['path']]
    status=out/'records/status.json';assert not status.exists(),'Existing attempt preserved'
    record=dict(status='running',case=a.case,argv=command,cwd=str(CODE),gpu=a.gpu,started_at=time.time())
    status.write_text(json.dumps(record,indent=2))
    with (out/'logs/train.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=proc.pid;status.write_text(json.dumps(record,indent=2));rc=proc.wait()
    record.update(exit_code=rc,seconds=time.time()-record['started_at'],status='complete' if rc==0 and (out/'final_scene.ply').exists() else 'failed')
    status.write_text(json.dumps(record,indent=2));print(record['status'],a.case,flush=True)
    raise SystemExit(rc if rc else (0 if record['status']=='complete' else 1))


if __name__=='__main__':main()
