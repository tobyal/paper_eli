"""Execute independent text localization then the existing GSLE generation stages."""
import argparse,json,os,subprocess,time,shutil
from pathlib import Path
from prepare import prepare,ROOT,BASE,CODE


def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--gpu',required=True);p.add_argument('--through',default='export');a=p.parse_args()
    prepare(a.case);out=BASE/'gsle_native'/a.case;native=BASE/'native_gsle'
    env=os.environ.copy()
    for key in ['TARGET_GEOMETRY','INSTANCE_SELECTION_LOG','GE_TARGET_CAMERA_DEPTH','GE_REUSE_ADD_ASSETS']:env.pop(key,None)
    env.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,GSLE_OUTPUT=str(out),GSLE_RECORDS=str(out/'records'),GSLE_CONFIG=str(out/'config.json'),
       HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
       OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTHONPATH=str(CODE),WANDB_MODE='disabled')
    stages=[('query','gsle_langsplat',native/'query_language.py',[]),('localization','gsle',native/'localize.py',[]),
      ('avp','gsle',native/'run_avp.py',[]),('inpainting','gsle',CODE/'inpaint.py',[]),('ranking','gsle_langsplat',CODE/'rank_inpainting.py',[]),
      ('foreground','gsle_langsplat',CODE/'sam_foreground.py',[]),('coarse','gsle',CODE/'train_object.py',['coarse','--microbatch','2']),
      ('composition','gsle',CODE/'compose.py',['coarse']),('refine','gsle',CODE/'train_object.py',['refine','--microbatch','4']),
      ('final_composition','gsle',CODE/'compose.py',['refine']),('export','gsle',native/'export.py',[])]
    stages=stages[:next(i for i,row in enumerate(stages) if row[0]==a.through)+1]
    path=out/'records/status.json';record=json.loads(path.read_text()) if path.exists() else dict(case=a.case,started_at=time.time(),stages={})
    for name,venv,script,extra in stages:
        if record['stages'].get(name,{}).get('status')=='complete':continue
        if record['stages'].get(name,{}).get('status')=='failed':raise RuntimeError('Previous failed attempt must be retained before explicit recovery')
        cmd=[str(ROOT/'envs'/venv/'bin/python'),str(script)]+extra
        record.update(status='running',active_stage=name,gpu=a.gpu)
        record['stages'][name]=dict(status='running',started_at=time.time(),argv=cmd);path.write_text(json.dumps(record,indent=2))
        with (out/'logs'/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=proc.pid;path.write_text(json.dumps(record,indent=2));rc=proc.wait()
        record['stages'][name].update(status='complete' if rc==0 else 'failed',exit_code=rc,seconds=time.time()-record['stages'][name]['started_at'])
        record['status']='running' if rc==0 else 'failed';path.write_text(json.dumps(record,indent=2))
        print(a.case,name,record['stages'][name]['status'],flush=True)
        if rc:raise SystemExit(rc)
    record.update(status='complete' if a.through=='export' else 'prepared_through_'+a.through,seconds=time.time()-record['started_at'])
    path.write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
