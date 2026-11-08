"""Read-only Gaussian inspection endpoints. Never start or alter experiments."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from . import store as s, service as v

router=APIRouter()


def selected_experiment(identifier=None):
    if identifier: return v.experiment(identifier)
    record=s.BASE/'records/multiview_scene_refinement_20261106.json'
    if record.exists(): return v.experiment(json.loads(record.read_text())['web_experiment'])
    for experiment in s.rows('SELECT * FROM experiments ORDER BY created_at DESC'):
        if s.unpack(experiment['heads']).get('m3'): return experiment
    raise HTTPException(404,'还没有可查看的 M3 设备')


def contained(path):
    path=Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (s.BASE/'outputs',s.DATA/'experiments')):
        raise HTTPException(404,'不允许的模型路径')
    if path.suffix.lower()!='.ply' or not path.is_file(): raise HTTPException(404,'高斯模型不存在')
    return path


def sources(experiment):
    heads=s.unpack(experiment['heads']); directories={k:v.stage_dir(heads[k]) for k in ('m1','m2','m3','m4') if k in heads}
    if not all(k in directories for k in ('m1','m2','m3')): raise HTTPException(404,'缺少设备或相机数据')
    candidates={'object_m3':directories['m3']/'object_world.ply',
                'background_m1':directories['m1']/'background_gaussians.ply'}
    if 'm4' in directories:
        if (directories['m4']/'object_world.ply').is_file():
            candidates['object_m3']=directories['m4']/'object_world.ply'
        fixed=directories['m4']/'object_fixed.ply'
        candidates['object_m4']=fixed if fixed.is_file() else directories['m4']/'object_refined.ply'
        bg=directories['m4']/'background_optimization/background_optimized.ply'
        if bg.is_file(): candidates['background_m4']=bg
    return {k:contained(p) for k,p in candidates.items() if p.is_file()},directories


@lru_cache(maxsize=16)
def geometry(path,size,mtime,up_tuple):
    data=PlyData.read(path,mmap='r')['vertex'].data
    xyz=np.column_stack([data[n] for n in ('x','y','z')]).astype(float)
    if not np.isfinite(xyz).all(): raise ValueError('设备包含无效坐标')
    low,high=xyz.min(0),xyz.max(0); center=(low+high)/2
    height=xyz@np.asarray(up_tuple); base=xyz[height<=np.quantile(height,.2)]
    base_low,base_high=base.min(0),base.max(0)
    return {'count':len(data),'center':center.tolist(),'radius':float(np.linalg.norm(high-low)/2),
            'base_center':((base_low+base_high)/2).tolist(),'base_radius':float(np.linalg.norm(base_high-base_low)/2)}


def count_vertices(path):
    with path.open('rb') as stream:
        for _ in range(100):
            line=stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '): return int(line.split()[-1])
            if line=='end_header': break
    raise ValueError('不是有效的 Gaussian PLY')


@router.get('/api/viewer/catalog')
def catalog(experiment_id: str=None):
    experiment=selected_experiment(experiment_id); paths,directories=sources(experiment)
    manifest=json.loads((directories['m1']/'manifest.json').read_text())
    anchor=json.loads((directories['m2']/'anchor_camera.json').read_text())
    up=np.asarray(manifest['world_up'],dtype=float); up/=np.linalg.norm(up)
    if 'object_m3' not in paths: raise HTTPException(409,'PBR M3 是局部设备；请完成 M4 空间对齐后查看场景漫游')
    obj=paths['object_m3']; stat=obj.stat()
    bounds=geometry(str(obj),stat.st_size,stat.st_mtime_ns,tuple(up))
    R=np.asarray(anchor['R']); t=np.asarray(anchor['t']); position=-R.T@t
    distance=np.linalg.norm(position-np.asarray(bounds['center']))
    return {'experiment':{'id':experiment['id'],'name':experiment['name']},'geometry':bounds,'up':up.tolist(),
        'anchor':{'position':position.tolist(),'target':(position+R[2]*distance).tolist(),
                  'fov':float(np.degrees(2*np.arctan(anchor['height']/(2*anchor['fy']))))},
        'resources':{key:{'url':f"/api/viewer/file/{experiment['id']}/{key}.ply?v={path.stat().st_mtime_ns}",
            'bytes':path.stat().st_size,'count':count_vertices(path)} for key,path in paths.items()},
        'source_unchanged':True,'renderer':'Spark 2.4.0 / Three.js 0.180.0'}


@router.get('/api/viewer/file/{experiment_id}/{resource_id}.ply')
def model_file(experiment_id: str,resource_id: str):
    paths,_=sources(selected_experiment(experiment_id))
    if resource_id not in paths: raise HTTPException(404,'未知的高斯素材')
    return FileResponse(paths[resource_id],media_type='application/octet-stream',
                        headers={'Cache-Control':'private, max-age=3600'})


@router.get('/viewer')
def viewer_page(): return FileResponse(Path(__file__).parent/'static/viewer.html')
