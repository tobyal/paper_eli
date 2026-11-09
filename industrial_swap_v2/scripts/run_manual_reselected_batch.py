"""Separate TRELLIS product-image inputs from full-canvas pose references."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import yaml
from PIL import Image
from industrial_swap_v2.core.context import Context, read_json, write_json, sha256
from industrial_swap_v2.scripts.run_pbr_case_batch import receipt_ready, gpu_available
from industrial_swap_v2.webapp import store as s, service as v

CASES=[('case_04_B','case04','灰白工业冷水机','case04_tre.png'),
       ('case_07_A','case07','蓝白零件存储柜','case07_tre.png'),
       ('case_10_C','case10','灰蓝工业工具柜','case10_pre.png')]


def prepare(batch,gpus):
    if (batch/'batch.json').exists():return
    source=s.BASE/'outputs/m2_reselected_04_07_10_v2';items=[]
    for i,(source_key,key,label,trellis) in enumerate(CASES):
        old=Context(source/source_key/'config.yaml');folder=batch/key
        folder.mkdir();(folder/'m1').symlink_to(old.stage('m1').resolve(),target_is_directory=True)
        shutil.copytree(old.stage('m2'),folder/'m2')
        for name in ('records','logs'):(folder/name).mkdir()
        for name in ('trellis_model.json','trellis_dependencies.json'):
            shutil.copy2(s.BASE/'records'/name,folder/'records'/name)
        cfg=copy.deepcopy(old.cfg);relative=str(folder.relative_to(s.BASE))
        cfg['project'].update(output_dir=relative,records_dir=relative+'/records',logs_dir=relative+'/logs',gpu=gpus[i%len(gpus)])
        cfg['m3']['input_image']=str((batch/'inputs'/trellis).relative_to(s.BASE))
        cfg['m3']['pbr'].pop('source_glb',None)
        cfg['m4'].pop('resume',None);cfg['m4']['integration']={'alignment':'rasterized_reference'}
        cfg['m4']['local_background']['iterations']=2000;cfg['m4']['save_every']=500
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        parent=read_json(s.BASE/'outputs/registration_all_cases_v1'/source_key/'web_registration.json')['experiment_id']
        heads=s.unpack(v.experiment(parent)['heads']);heads={k:value for k,value in heads.items() if k=='m1'}
        scene=batch/'inputs'/(key+'.png');product=batch/'inputs'/trellis
        with Image.open(scene) as image:
            camera=read_json(folder/'m2/anchor_camera.json')
            assert abs(image.width/image.height-camera['width']/camera['height'])<1e-6
        write_json(folder/'source_preservation.json',{str(p.resolve()):sha256(p) for p in (
            scene,product,old.stage('m1')/'background_gaussians.ply',old.stage('m2')/'anchor_camera.json',old.stage('m2')/'generation_region.json')})
        write_json(folder/'provenance.json',{'source_experiment':parent,'source_heads':heads,
            'prepared_m2':str(old.stage('m2')),'full_scene_reference':str(scene),'trellis_input':str(product),
            'trellis_input_sha256':sha256(product),'qwen_calls':0,'new_trellis_calls':1,
            'direct_product_input':True,'pose_reference':'SAM foreground of user-generated full scene on new M2 Anchor',
            'case10_pre_filename_used':key=='case10','prior_results_overwritten':False})
        items.append({'key':key,'label':label,'folder':str(folder),'experiment_id':parent,'register_new_m2':True,
            'reuse_pbr_m3':False,'scene_input':str(scene),'trellis_input':str(product),'status':'pending','stages':{}})
    write_json(batch/'batch.json',{'name':batch.name,'items':items,'gpus':gpus,'qwen_calls':0,
        'new_trellis_inputs':True,'m4_iterations':2000,'policy':'New views; direct user product images for TRELLIS; full scene images for alignment.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);p.add_argument('--gpus',type=int,nargs='+',default=[2,3,0])
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();batch=a.batch.resolve();batch.relative_to(s.BASE/'outputs')
    prepare(batch,a.gpus)
    if a.prepare_only:return
    with (batch/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=read_json(batch/'batch.json');active={}
        for item in state['items']:
            if item['status']=='running':item['status']='pending'
        def publish():
            with (batch/'gallery.log').open('a') as log:
                subprocess.run([str(s.BASE/'envs/webapp/bin/python'),'-m','industrial_swap_v2.scripts.publish_pbr_case_batch',
                    '--batch',str(batch)],cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        while True:
            changed=False
            for gpu,(process,item,stage,started,log) in list(active.items()):
                if process.poll() is None:continue
                log.close();del active[gpu];folder=Path(item['folder'])
                item['stages'][stage]={'exit_code':process.returncode,'seconds':time.monotonic()-started,'gpu':gpu}
                if process.returncode or (stage!='export' and not receipt_ready(folder,stage)):
                    item.update(status='failed',failed_stage=stage)
                elif stage=='export':item['status']='completed'
                else:item['status']='pending'
                changed=True;print(item['key'],stage,item['status'],flush=True)
            for gpu in a.gpus:
                if gpu in active:continue
                chosen=None
                for candidate in state['items']:
                    if candidate['status']!='pending':continue
                    folder=Path(candidate['folder']);stage=next((k for k in ('m2','m3','m4') if not receipt_ready(folder,k)),'export')
                    cfg=yaml.safe_load((folder/'config.yaml').read_text())
                    needed=max(20,cfg['m3']['minimum_free_gpu_gib']+1) if stage=='m3' else 28 if stage=='m4' else 8
                    if gpu_available(gpu,needed):chosen=(candidate,folder,stage);break
                if chosen is None:continue
                item,folder,stage=chosen
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),SWAP_GPU=str(gpu),ROOT=str(s.ROOT),
                    OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1',SWAP_DISABLE_QWEN='1')
                module={'m2':'scripts.import_case_image','m3':'modules.m3_trellis_generation',
                    'm4':'modules.m4_refinement','export':'scripts.render_pbr_scene_results'}[stage]
                command=[str(s.ROOT/'envs/gsle/bin/python'),'-m','industrial_swap_v2.'+module,'--config',str(folder/'config.yaml')]
                if stage=='m2':command+=['--image',item['scene_input']]
                log=(folder/'logs'/f'{stage}.log').open('a')
                process=subprocess.Popen(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item.update(status='running',active_stage=stage,gpu=gpu,pid=process.pid)
                active[gpu]=(process,item,stage,time.monotonic(),log);changed=True
                print(item['key'],'started',stage,'GPU',gpu,flush=True)
            if changed:write_json(batch/'batch.json',state);publish()
            if not active and all(x['status'] in ('completed','failed') for x in state['items']):break
            time.sleep(8)
        if any(x['status']=='failed' for x in state['items']):raise SystemExit(1)


if __name__=='__main__':main()
