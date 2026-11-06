"""Exercise real GLB upload, local PLY alignment and world PLY composition via HTTP."""
import json
from pathlib import Path
import time
import uuid
import httpx
import numpy as np
from plyfile import PlyData
from . import store as s
from . import service as v


def main():
    report_path=s.DATA/'validation/asset_routes.json'; report={}
    if report_path.exists(): report=json.loads(report_path.read_text())
    client=httpx.Client(base_url='http://127.0.0.1:18086/api',headers={'X-Swap-Action':'1'},timeout=180,trust_env=False)
    def get(path):
        r=client.get(path); r.raise_for_status(); return r.json()
    def post(path,body):
        r=client.post(path,json=body); r.raise_for_status(); return r.json()
    def save(): report_path.write_text(s.dump(report),encoding='utf-8')
    baseline=next(e for e in get('/experiments') if e['name']=='原始基线 · 黄色机械臂')
    assets=baseline['head_runs']['m3']['outputs']['assets']
    if 'glb_upload' not in report:
        with (s.BASE/'outputs/m3/trellis_asset.glb').open('rb') as file:
            r=client.post('/assets/upload',data={'kind':'mesh','name':'验证上传 · 原始 GLB','model':'TRELLIS.2 upload fixture'},
                files={'file':('robot.glb',file,'model/gltf-binary')})
        r.raise_for_status(); report['glb_upload']=r.json()['id']; report['glb_deduplicated']=r.json()['deduplicated']; save()
    for key,kind,realign in [('local','object_local.ply',True),('world','object_world.ply',False)]:
        if key not in report:
            exp=post('/experiments',{'name':'Web 路径验证 · '+key+' PLY','copy_from':baseline['id']})
            post('/experiments/'+exp['id']+'/selection',{'role':'m3','asset_id':assets[kind]})
            run=post('/experiments/'+exp['id']+'/runs',{'stage':'m3','action':'reuse',
                'request_id':str(uuid.uuid4()),'parameters':{'realign':realign}})
            report[key]={'experiment':exp['id'],'run':run['id']}; save()
    for key in ('local','world'):
        previous=None
        while True:
            run=get('/runs/'+report[key]['run'])
            if run['status']!=previous: print(key,run['status'],flush=True); previous=run['status']
            if run['status']=='failed': raise RuntimeError(run['error'])
            if run['status']=='completed':
                directory=v.stage_dir(run['id'])
                proof=json.loads((directory/'reuse.json').read_text())
                assert not proof['trellis_called']
                name='object_local.ply' if key=='local' else 'object_world.ply'
                original=PlyData.read(str(s.asset_path(assets[name])))['vertex'].data
                imported=PlyData.read(str(directory/name))['vertex'].data
                preserved=len(original)==len(imported) and all(np.array_equal(original[n],imported[n]) for n in original.dtype.names)
                assert preserved, 'Imported Gaussian attributes changed unexpectedly'
                report[key].update(status='complete',trellis_called=False,imported_gaussian_attributes_preserved=preserved,
                    points=len(imported),output_integrity_verified=v.verify_outputs(v.run(run['id'])))
                save(); break
            time.sleep(5)
    report['status']='complete'; report['completed_at']=s.now(); save(); print('ASSET ROUTES COMPLETE',flush=True)


if __name__=='__main__': main()
