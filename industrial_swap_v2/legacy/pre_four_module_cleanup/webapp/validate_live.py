"""Real server end-to-end validation. Never invokes the Qwen action.

Uploads the already available edited industrial image as an external-upload fixture,
runs SAM, explicitly exercises one fresh official TRELLIS inference, then reuses
that GLB in another experiment and runs the configured 2,000-step refinement.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import zipfile
import io
import httpx
from . import store as s


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume-failed',action='store_true'); args=parser.parse_args()
    output=s.DATA/'validation'; output.mkdir(exist_ok=True)
    report_path=output/'live_run.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'started_at':s.now(),'runs':{},'requests':{}}
    def save(): report_path.write_text(s.dump(report),encoding='utf-8')
    client=httpx.Client(base_url='http://127.0.0.1:18086/api',headers={'X-Swap-Action':'1'},timeout=180,trust_env=False)
    def get(path):
        response=client.get(path); response.raise_for_status(); return response.json()
    def post(path, body):
        response=client.post(path,json=body)
        if response.is_error: raise RuntimeError(response.text)
        return response.json()
    def wait(key,eid,stage,action,parameters=None):
        if key not in report['runs']:
            request_id=report['requests'].setdefault(key,str(uuid.uuid4())); save()
            value=post('/experiments/'+eid+'/runs',{'stage':stage,'action':action,'request_id':request_id,'parameters':parameters or {}})
            report['runs'][key]=value['id']; save()
        rid=report['runs'][key]; previous=None
        value=get('/runs/'+rid)
        if value['status']=='failed' and args.resume_failed: post('/runs/'+rid+'/resume',{})
        while True:
            value=get('/runs/'+rid)
            if value['status']!=previous: print(key,rid,value['status'],flush=True); previous=value['status']
            if value['status']=='completed': return value
            if value['status']=='failed': raise RuntimeError(key+': '+str(value['error']))
            time.sleep(5)
    if 'baseline_hashes' not in report:
        report['baseline_hashes']={}
        for stage in ('m1','m2','m3','m4'):
            receipt=json.loads((s.BASE/'records'/f'{stage}_status.json').read_text())
            for artifact in receipt['artifacts']: report['baseline_hashes'][artifact['path']]=s.sha(artifact['path'])
        save()
    if 'experiment_a' not in report:
        report['experiment_a']=post('/experiments',{'name':'Web 验证 A · 上传图像与真实三维生成'})['id']; save()
    a=report['experiment_a']; wait('m1_a',a,'m1','reuse'); material=wait('materials_a',a,'m2','prepare')
    response=client.get('/runs/'+material['id']+'/bundle'); response.raise_for_status()
    names=zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert {'anchor.png','position_hint.png','edit_mask.png','depth.png','depth_background.png','generation_prompt.txt','anchor_camera.json','conditions.json'}.issubset(names)
    (output/'generation_materials.zip').write_bytes(response.content); report['materials_zip']=names; save()
    if 'uploaded_image' not in report:
        with (s.BASE/'outputs/m2/edited_scene.png').open('rb') as image:
            response=client.post('/assets/upload',data={'kind':'edited_scene','name':'验证上传 · 原有机械臂场景图',
                'experiment_id':a,'camera_aligned':'true','model':'existing Qwen output used as upload fixture'},
                files={'file':('external_fixture.png',image,'image/png')})
        response.raise_for_status(); asset=response.json(); report['uploaded_image']=asset['id']; report['upload_deduplicated']=asset['deduplicated']; save()
        post('/experiments/'+a+'/selection',{'asset_id':asset['id'],'role':'m2'})
    wait('m2_a',a,'m2','reuse')
    m3=wait('m3_a',a,'m3','trellis',{'force_model':True})
    model=m3['outputs']['assets']['trellis_asset.glb']; report['generated_model']=model; save()
    if 'experiment_b' not in report:
        report['experiment_b']=post('/experiments',{'name':'Web 验证 B · 三维素材复用与 DI-SDS','copy_from':a})['id']; save()
        post('/experiments/'+report['experiment_b']+'/selection',{'asset_id':model,'role':'m3'})
    b=report['experiment_b']; reused=wait('m3_b',b,'m3','reuse',{'realign':True})
    refined=wait('m4_b',b,'m4','refine',{'iterations':2000})
    # Repeating the exact stage should reuse immutable outputs, not rerun TRELLIS or alignment.
    if 'experiment_c' not in report:
        report['experiment_c']=post('/experiments',{'name':'Web 验证 C · 相同输入缓存','copy_from':b})['id']; save()
    repeated=wait('m3_cache',report['experiment_c'],'m3','reuse',{'realign':True})
    if not repeated['reused_from']:
        # A code/signature update during development can require one new cache entry.
        # Repeating that exact immutable-input run must then be a cache hit.
        repeated=wait('m3_cache_repeat',report['experiment_c'],'m3','reuse',{'realign':True})
    report['stage_cache_verified']=bool(repeated['reused_from'])
    if 'experiment_d' not in report:
        report['experiment_d']=post('/experiments',{'name':'Web 验证 D · TRELLIS 自动模型缓存','copy_from':a})['id']; save()
    cached_model=wait('m3_model_cache',report['experiment_d'],'m3','trellis')
    response=client.get('/runs/'+cached_model['id']+'/file',params={'name':'reuse.json'}); response.raise_for_status()
    proof=response.json()
    report['model_cache_verified']=proof['automatic_generation_cache'] and not proof['trellis_called']
    report['historical_results_unchanged']=all(s.sha(path)==old for path,old in report['baseline_hashes'].items())
    report['new_qwen_submissions']=len(list((s.DATA/'experiments').glob('*/runs/*/records/qwen_generation_once.lock')))
    valid=report['historical_results_unchanged'] and report['new_qwen_submissions']==0 and report['stage_cache_verified'] and report['model_cache_verified']
    report['completed_at']=s.now(); report['status']='complete' if valid else 'failed'; save()
    assert valid
    print('LIVE VALIDATION COMPLETE',str(report_path),flush=True)


if __name__=='__main__': main()
