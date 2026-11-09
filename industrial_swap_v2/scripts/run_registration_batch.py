"""Apply the tested case02 registration + 2000-step M4 to remaining cases."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
from industrial_swap_v2.core.context import read_json,write_json
from industrial_swap_v2.webapp import store as s
from industrial_swap_v2.scripts.run_pbr_case_batch import receipt_ready,gpu_available


def prepare(batch,gpus):
    if (batch/'batch.json').exists():return
    if batch.exists():raise ValueError('Choose a new batch directory')
    batch.mkdir(parents=True)
    old=read_json(s.BASE/'outputs/pbr_10cases_m4_2000_20261108/batch.json');items=[]
    for i,item in enumerate(old['items']):
        key=item['key'];reused=key=='case_02_B'
        folder=s.BASE/'outputs/case02_raster_alignment_v1' if reused else batch/key
        if not reused:
            subprocess.run([str(s.BASE/'envs/webapp/bin/python'),'-m','industrial_swap_v2.scripts.prepare_reference_alignment',
                '--source-config',str(Path(item['folder'])/'config.yaml'),'--destination',str(folder.relative_to(s.BASE)),
                '--gpu',str(gpus[i%len(gpus)]),'--iterations','2000'],cwd=s.ROOT,check=True)
        items.append({'key':key,'label':item['label'],'folder':str(folder),'reused_case02':reused,
            'status':'completed' if reused else 'pending','stages':{}})
    write_json(batch/'batch.json',{'name':batch.name,'gpus':gpus,'items':items,'m4_steps':2000,
        'qwen_calls':0,'trellis_calls':0,'new_m3_training':False,'created_at':s.now()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);p.add_argument('--gpus',nargs='+',type=int,default=[2,3,0])
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();batch=a.batch.resolve();batch.relative_to(s.BASE/'outputs')
    prepare(batch,a.gpus)
    if a.prepare_only:return
    with (batch/'scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=read_json(batch/'batch.json');active={}
        for item in state['items']:
            if item['status']=='running':item['status']='pending'
        def gallery():
            with (batch/'gallery.log').open('a') as log:
                subprocess.run([str(s.BASE/'envs/webapp/bin/python'),'-m','industrial_swap_v2.scripts.publish_registration_batch',
                    '--batch',str(batch)],cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(batch/'batch.json',state);gallery()
        while True:
            changed=False
            for gpu,(process,item,stage,started,log) in list(active.items()):
                if process.poll() is None:continue
                log.close();del active[gpu]
                item['stages'][stage]={'exit_code':process.returncode,'seconds':time.monotonic()-started,'gpu':gpu}
                if process.returncode or (stage=='m4' and not receipt_ready(Path(item['folder']),'m4')):
                    item.update(status='failed',failed_stage=stage)
                elif stage=='inspect':
                    folder=Path(item['folder'])
                    with (folder/'logs/publish.log').open('a') as log:
                        result=subprocess.run([str(s.BASE/'envs/webapp/bin/python'),'-m','industrial_swap_v2.scripts.publish_reference_alignment',
                            '--config',str(folder/'config.yaml'),'--label',item['key']+' · '+item['label'],
                            '--page-name',batch.name+'_'+item['key']],cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT)
                    item.update(status='completed' if result.returncode==0 else 'failed')
                    if result.returncode:item['failed_stage']='publish'
                else:item['status']='pending'
                print(item['key'],stage,item['status'],flush=True);changed=True
            for gpu in a.gpus:
                if gpu in active:continue
                if not gpu_available(gpu,28):continue
                item=next((x for x in state['items'] if x['status']=='pending'),None)
                if item is None:break
                folder=Path(item['folder']);stage='inspect' if receipt_ready(folder,'m4') else 'm4'
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),SWAP_GPU=str(gpu),ROOT=str(s.ROOT),
                    OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',SWAP_DISABLE_QWEN='1')
                module='modules.m4_refinement' if stage=='m4' else 'scripts.inspect_reference_alignment'
                command=[str(s.ROOT/'envs/gsle/bin/python'),'-m','industrial_swap_v2.'+module,'--config',str(folder/'config.yaml')]
                log=(folder/'logs'/f'{stage}.log').open('a')
                process=subprocess.Popen(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item.update(status='running',active_stage=stage,pid=process.pid,gpu=gpu)
                active[gpu]=(process,item,stage,time.monotonic(),log);changed=True
                print(item['key'],'started',stage,'GPU',gpu,flush=True)
            if changed:write_json(batch/'batch.json',state);gallery()
            if not active and all(x['status'] in ('completed','failed') for x in state['items']):break
            time.sleep(8)
        write_json(batch/'batch.json',state);gallery()
        if any(x['status']=='failed' for x in state['items']):raise SystemExit(1)


if __name__=='__main__':main()
