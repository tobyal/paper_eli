"""Independent single-GPU pose review/M4 worker, using existing generated assets.

The Web worker continues TRELLIS jobs. This process never claims Web pending jobs;
it imports only completed, SHA-verified CLI outputs after explicit pose review.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
import yaml
from industrial_swap_v2.core.context import read_json,write_json
from industrial_swap_v2.webapp import store as s,service as v


def register_cli(item,stage,folder,seconds,batch_name):
    request=batch_name+':'+item['key']+':reviewed-cli-'+stage
    previous=s.one('SELECT * FROM stage_runs WHERE request_id=:r',r=request)
    if previous and previous['status']=='completed':return previous['id']
    receipt=read_json(folder/'records'/f'{stage}_status.json')
    if receipt['status']!='complete' or not all(s.sha(x['path'])==x['sha256'] for x in receipt['artifacts']):
        raise ValueError('CLI receipt integrity failed')
    exp=v.experiment(item['experiment_id']);heads=s.unpack(exp['heads'])
    inputs={k:heads[k] for k in (('m1','m2') if stage=='m3' else ('m1','m2','m3'))}
    rid=previous['id'] if previous else s.uid()
    log=folder/(stage+'.log')
    if not previous:
        with s.transaction() as conn:
            v.active(conn,item['experiment_id'])
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
                config_path,folder,log_path,created_at,started_at,request_id)
                VALUES(:id,:e,:stage,:action,'importing',:sig,:inputs,:params,:cfg,:folder,:log,:t,:t,:req)''',
                id=rid,e=item['experiment_id'],stage=stage,action='reuse' if stage=='m3' else 'refine',
                sig=s.digest({'verified_cli_receipt':receipt,'inputs':inputs}),inputs=s.dump(inputs),
                params=s.dump({'realign':True} if stage=='m3' else {'iterations':250}),
                cfg=str(folder/'config.yaml'),folder=str(folder),log=str(log),t=s.now(),req=request)
            s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,external_api,status,created_at)
                VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'importing',:t)''',id=s.uid(),e=item['experiment_id'],r=rid,
                model='TRELLIS.2 reused asset, RGB pose fit' if stage=='m3' else 'dual_mask_local_background',
                prompt=item['description'],inputs=s.dump(inputs),t=s.now())
    v.finish(rid,folder/stage,seconds=seconds)
    return rid


def gpu_ready(gpu):
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in value.splitlines():
        index,free,load=map(int,line.split(','))
        if index==gpu:return free>=24*1024 and load<30
    return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);p.add_argument('--gpu',type=int,default=3)
    args=p.parse_args();batch=args.batch.resolve();batch.relative_to(s.BASE/'outputs')
    lock=(batch/'postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    file=batch/'postprocess.json';state=read_json(file) if file.exists() else {'items':{},'gpu':args.gpu}
    env=os.environ.copy();env.update(SWAP_GPU=str(args.gpu),CUDA_VISIBLE_DEVICES=str(args.gpu),
        SWAP_DISABLE_QWEN='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',
        ROOT=str(s.ROOT),CUDA_HOME='/usr/local/cuda-11.8')
    env['PATH']='/usr/local/cuda-11.8/bin:'+env['PATH']
    python=str(s.ROOT/'envs/gsle/bin/python')
    while True:
        manifest=read_json(batch/'manifest.json');did_work=False
        for item in manifest['items']:
            key=item['key'];current=state['items'].setdefault(key,{'status':'waiting_m3'})
            if current['status'] in ('completed','failed'):continue
            source_id=item['runs'].get('m3')
            if not source_id or v.run(source_id)['status']!='completed':continue
            if not gpu_ready(args.gpu):continue
            folder=batch/'aligned'/key;receipt=folder/'records/m3_status.json'
            try:
                if not receipt.exists():
                    current.update(status='fitting_reference',source_m3_run=source_id);write_json(file,state)
                    command=[python,'-m','industrial_swap_v2.scripts.refit_m3_reference',
                        '--source-config',v.run(source_id)['config_path'],'--destination',str(folder.relative_to(s.BASE)),
                        '--gpu',str(args.gpu)]
                    folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'m3.log').open('a') as log:
                        started=time.monotonic();subprocess.run(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    current['m3_seconds']=time.monotonic()-started
                    print(key,'reference alignment ready for visual review',flush=True)
                    did_work=True
                review=batch/'reviews'/(key+'.json')
                if not review.exists() or not read_json(review).get('approved_for_m4'):
                    current['status']='awaiting_pose_review';write_json(file,state);continue
                if read_json(review)['object_world_sha256']!=s.sha(folder/'m3/object_world.ply'):
                    raise ValueError('Reviewed object changed')
                current['m3_run']=register_cli(item,'m3',folder,current.get('m3_seconds'),batch.name)
                if not (folder/'records/m4_status.json').exists():
                    current['status']='m4_optimizing';write_json(file,state)
                    with (folder/'m4.log').open('a') as log:
                        started=time.monotonic();subprocess.run([python,'-m','industrial_swap_v2.modules.m4_refinement',
                            '--config',str(folder/'config.yaml')],cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    current['m4_seconds']=time.monotonic()-started
                current['m4_run']=register_cli(item,'m4',folder,current.get('m4_seconds',0),batch.name)
                current.update(status='completed',folder=str(folder));write_json(file,state)
                print(key,'M4 completed and registered in Web',flush=True);did_work=True
            except Exception as exc:
                current.update(status='failed',error=str(exc),traceback=traceback.format_exc());write_json(file,state)
                print(key,'FAILED',traceback.format_exc(),flush=True)
        write_json(file,state)
        if len(state['items'])==len(manifest['items']) and all(x['status'] in ('completed','failed') for x in state['items'].values()):return
        if not did_work:time.sleep(10)


if __name__=='__main__':main()
