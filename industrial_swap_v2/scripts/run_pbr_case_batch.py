"""Update the ten existing devices using verified PBR M3 and 2000-step M4.

Reuses all GLBs/placements and case00's completed PBR M3. Each case writes only
to its new directory. One process per selected GPU; never claims Web jobs.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import yaml
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256
from industrial_swap_v2.webapp import service as v,store as s


def prepare(batch,gpus):
    if (batch/'batch.json').exists():return
    if batch.exists():raise ValueError('Batch directory already exists without a manifest')
    batch.mkdir(parents=True)
    defaults=yaml.safe_load((s.BASE/'configs/default.yaml').read_text())
    source=read_json(s.BASE/'outputs/qwen9_completion_20261107/final_manifest.json')
    robot=s.BASE/'outputs/pbr_case00_20261108';robot_web=read_json(robot/'web_registration.json')
    entries=[{'key':'case00','label':'黄色六轴工业机械臂','experiment_id':robot_web['experiment_id'],
        'source_config':robot/'config.yaml','source_m3':robot/'m3','reuse_pbr_m3':True,
        'reuse_m3_run':robot_web['runs']['m3']},
        *[{'key':item['key'],'label':item['label'],'experiment_id':item['experiment_id'],
            'source_config':Path(v.run(item['states']['m3']['id'])['config_path']),
            'source_m3':Path(item['states']['m3']['output']),'reuse_pbr_m3':False} for item in source['items']]]
    items=[]
    for index,entry in enumerate(entries):
        folder=batch/entry['key'];folder.mkdir()
        cfg=yaml.safe_load(entry['source_config'].read_text());heads=s.unpack(v.experiment(entry['experiment_id'])['heads'])
        for name in ('m1','m2'):(folder/name).symlink_to(v.stage_dir(heads[name]),target_is_directory=True)
        for name in ('logs','records'):(folder/name).mkdir()
        relative=str(folder.relative_to(s.BASE));cfg['project'].update(output_dir=relative,
            records_dir=relative+'/records',logs_dir=relative+'/logs',gpu=gpus[index%len(gpus)])
        cfg['m3'].update(method='pbr_multiview_gaussian',input_image=relative+'/m2/object_rgba.png',
            pbr=copy.deepcopy(defaults['m3']['pbr']))
        cfg['m3']['pbr']['source_glb']=str(entry['source_m3']/'trellis_asset.glb')
        if entry['reuse_pbr_m3']:
            old=yaml.safe_load(entry['source_config'].read_text())
            cfg['m3']['pbr']=copy.deepcopy(old['m3']['pbr'])
            transform=Path(old['m4']['integration']['existing_transform'])
            (folder/'m3').symlink_to(entry['source_m3'],target_is_directory=True)
            (folder/'m4').mkdir()
            (folder/'m4/multiview_supervision').symlink_to(robot/'m4/multiview_supervision',target_is_directory=True)
        else:transform=entry['source_m3']/'transform.json'
        cfg['m4'].pop('resume',None)
        cfg['m4'].update(method='dual_mask_local_background',integration={
            'alignment':'verified_transform','existing_transform':str(transform)},save_every=500)
        cfg['m4']['local_background']['iterations']=2000
        (folder/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        sources=[entry['source_m3']/'trellis_asset.glb',transform,folder/'m1/background_gaussians.ply',
            folder/'m1/target_gaussians.ply',folder/'m2/object_rgba.png',folder/'m2/object_mask.png',folder/'m2/anchor_camera.json']
        write_json(folder/'source_preservation.json',{str(p.resolve()):sha256(p) for p in sources})
        write_json(folder/'provenance.json',{'case':entry['key'],'source_experiment':entry['experiment_id'],
            'source_m3':str(entry['source_m3']),'source_heads':heads,'new_trellis_calls':0,'qwen_calls':0,
            'prior_results_overwritten':False,'placement_reused':str(transform)})
        if entry['reuse_pbr_m3']:
            ctx=Context(folder/'config.yaml');old_receipt=read_json(robot/'records/m3_status.json')
            ctx.complete('m3',[a['path'] for a in old_receipt['artifacts']],method='pbr_multiview_gaussian',
                reused_from=str(robot/'m3'),training_steps=4000,best_step=2250,new_training_steps=0)
            (folder/'logs/m3.log').write_text('Reused SHA-verified completed case00 PBR M3; no retraining.\n')
        items.append({**{k:str(value) if isinstance(value,Path) else value for k,value in entry.items()},
            'folder':str(folder),'status':'pending','m3_steps':4000,'m4_steps':2000,'stages':{}})
    write_json(batch/'batch.json',{'name':batch.name,'created_at':s.now(),'gpus':gpus,'items':items,
        'qwen_calls':0,'trellis_calls':0,'m4_iterations':2000,'policy':'Existing PBR method, existing GLBs/poses, isolated outputs.'})


def receipt_ready(folder,stage):
    receipt=folder/'records'/f'{stage}_status.json'
    return receipt.exists() and read_json(receipt).get('status')=='complete'


def gpu_available(gpu,needed):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in lines.splitlines():
        index,free,util=map(int,line.split(','))
        if index==gpu:return free>=needed*1024 and util<40
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True,type=Path)
    parser.add_argument('--gpus',nargs='+',type=int,default=[2,3,0]);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();batch=args.batch.resolve();batch.relative_to(s.BASE/'outputs')
    prepare(batch,args.gpus)
    if args.prepare_only:return
    with (batch/'scheduler.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read_json(batch/'batch.json');active={}
        # Completed receipts make process restarts inexpensive; never rerun a completed stage.
        for item in state['items']:
            if item['status']=='running':item['status']='pending'
        def publish():
            with (batch/'gallery.log').open('a') as log:
                subprocess.run([str(s.BASE/'envs/webapp/bin/python'),'-m',
                    'industrial_swap_v2.scripts.publish_pbr_case_batch','--batch',str(batch)],
                    cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(batch/'batch.json',state);publish()
        while True:
            changed=False
            for gpu,value in list(active.items()):
                process,item,stage,started,log=value
                if process.poll() is None:continue
                log.close();del active[gpu];folder=Path(item['folder'])
                item['stages'][stage]={'exit_code':process.returncode,'seconds':time.monotonic()-started,'gpu':gpu}
                if process.returncode or (stage in ('m3','m4') and not receipt_ready(folder,stage)):
                    item.update(status='failed',failed_stage=stage)
                elif stage=='export':item.update(status='completed',completed_at=s.now())
                else:item['status']='pending'
                changed=True;print(item['key'],stage,item['status'],flush=True)
            for gpu in args.gpus:
                if gpu in active:continue
                chosen=None
                for item in state['items']:
                    if item['status']!='pending':continue
                    folder=Path(item['folder'])
                    stage='m3' if not receipt_ready(folder,'m3') else 'm4' if not receipt_ready(folder,'m4') else 'export'
                    if (folder/'previews/render_receipt.json').exists():
                        item['status']='completed';changed=True;continue
                    required=28 if stage=='m4' else 12
                    if gpu_available(gpu,required):chosen=(item,stage);break
                if chosen is None:continue
                item,stage=chosen;folder=Path(item['folder']);env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),SWAP_GPU=str(gpu),ROOT=str(s.ROOT),OMP_NUM_THREADS='8',
                    OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',SWAP_DISABLE_QWEN='1')
                module={'m3':'modules.m3_trellis_generation','m4':'modules.m4_refinement','export':'scripts.render_pbr_scene_results'}[stage]
                command=[str(s.ROOT/'envs/gsle/bin/python'),'-m','industrial_swap_v2.'+module,'--config',str(folder/'config.yaml')]
                log=(folder/'logs'/f'{stage}.log').open('a')
                process=subprocess.Popen(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item.update(status='running',active_stage=stage,pid=process.pid,gpu=gpu)
                active[gpu]=(process,item,stage,time.monotonic(),log);changed=True
                print(item['key'],'started',stage,'GPU',gpu,flush=True)
            if changed:
                write_json(batch/'batch.json',state);publish()
            if not active and all(i['status'] in ('completed','failed') for i in state['items']):break
            time.sleep(8)
        write_json(batch/'batch.json',state);publish()
        if any(i['status']=='failed' for i in state['items']):raise SystemExit(1)


if __name__=='__main__':main()
