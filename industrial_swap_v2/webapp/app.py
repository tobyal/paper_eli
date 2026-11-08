"""Loopback-only FastAPI application; private models/credentials never reach clients."""
from pathlib import Path
import json
import mimetypes
from urllib.parse import urlsplit
from typing import Optional
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import store as s
from . import service as v

s.init()
app=FastAPI(title='Industrial Swap Lab',version='1.0',docs_url='/api/docs')
from .gaussian_viewer import router as viewer_router
app.include_router(viewer_router)
from . import m1_cases
m1_cases.init()
app.include_router(m1_cases.router)
from . import m2_cases
m2_cases.init()
app.include_router(m2_cases.router)
STATIC=Path(__file__).parent/'static'


@app.middleware('http')
async def local_mutations(request: Request, call_next):
    if request.method not in ('GET','HEAD','OPTIONS'):
        if request.headers.get('X-Swap-Action')!='1':
            return JSONResponse({'detail':'缺少同源操作标记'},status_code=403)
        origin=request.headers.get('origin')
        if origin and urlsplit(origin).netloc!=request.headers.get('host'):
            return JSONResponse({'detail':'不接受跨站写入请求'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    return response


@app.exception_handler(ValueError)
async def invalid(_, exc): return JSONResponse({'detail':str(exc)},status_code=422)


class NewExperiment(BaseModel):
    name: str=Field(min_length=1,max_length=120)
    replacement_text: Optional[str]=Field(default=None,max_length=3000)
    copy_from: Optional[str]=None

class Selection(BaseModel):
    asset_id: str
    role: str

class StageRequest(BaseModel):
    stage: str
    action: str
    request_id: str
    parameters: dict=Field(default_factory=dict)
    confirm_qwen: bool=False


def public_run(value):
    value=dict(value); folder=Path(value['folder']); value.pop('config_path',None); value.pop('folder',None); value.pop('log_path',None)
    for key in ('inputs','parameters','outputs'): value[key]=s.unpack(value[key])
    value['outputs'].pop('directory',None)
    value['outputs'].pop('integrity',None)
    progress=folder/'outputs/m4/background_optimization/training.jsonl'
    if value['stage']=='m4' and progress.exists():
        with progress.open('rb') as stream:
            stream.seek(max(0,progress.stat().st_size-8192)); lines=stream.read().splitlines()
        for line in reversed(lines):
            try: saved=json.loads(line)
            except (ValueError,UnicodeDecodeError): continue
            value['progress']={'step':saved['step'],'total':value['parameters'].get('iterations',250)}
            break
    else:
        progress=folder/'records/m4_progress.json'
        if value['stage']=='m4' and progress.exists():
            saved=json.loads(progress.read_text())
            value['progress']={'step':saved['step'],'total':value['parameters'].get('iterations',2000),
                'gpu_peak_gib':saved.get('gpu_peak_gib'),'historical':True}
    return value


def public_experiment(value):
    value=dict(value); cfg=s.unpack(value.pop('config')); value['heads']=s.unpack(value['heads'])
    from ..core.workflow import current_config
    execution=current_config(cfg)
    value['settings']={'iterations':execution['m4']['local_background']['iterations'],
        'save_every':execution['m4'].get('save_every',0),'scene_path':cfg['project']['source_ply'],
        'localization':cfg['m1']['localization_dir'],'anchor':cfg['m2']['existing_anchor'],
        'm2_model':cfg['m2']['model'],'m3_model':cfg['m3']['model'],
        'm3_method':cfg['m3'].get('method','legacy_mesh_sampling'),
        'confirmed_m1_case_id':cfg['m2'].get('confirmed_m1_case_id'),
        'preparation_only':bool(cfg['m2'].get('preparation_only')),
        'dual_mask_supervision':True,
        'local_background':True,'method':execution['m4']['method']}
    value['runs']=[public_run(r) for r in s.rows('SELECT * FROM stage_runs WHERE experiment_id=:e ORDER BY created_at DESC',e=value['id'])]
    value['selected']={role:v.selected(value['id'],role) for role in ('m2','m3')}
    # Cloned experiments can reference completed immutable stages of another experiment.
    value['head_runs']={key:public_run(v.run(rid)) for key,rid in value['heads'].items()}
    return value


@app.get('/api/health')
def health(): return {'status':'ok','database':'sqlite','qwen_on_navigation':False}


@app.get('/api/experiments')
def experiments(): return [public_experiment(r) for r in s.rows('SELECT * FROM experiments ORDER BY created_at DESC')]


@app.post('/api/experiments')
def new_experiment(body: NewExperiment):
    return public_experiment(v.experiment(v.create(body.name,body.replacement_text,body.copy_from)))


@app.get('/api/experiments/{identifier}')
def experiment(identifier: str): return public_experiment(v.experiment(identifier))


@app.post('/api/experiments/{identifier}/selection')
def selection(identifier: str, body: Selection):
    v.select(identifier,body.asset_id,body.role)
    return public_experiment(v.experiment(identifier))


@app.post('/api/experiments/{identifier}/runs')
def submit(identifier: str, body: StageRequest):
    v.experiment(identifier)
    rid=v.submit(identifier,body.stage,body.action,body.request_id,body.parameters,body.confirm_qwen)
    return public_run(v.run(rid))


@app.get('/api/runs')
def runs(): return [public_run(r) for r in s.rows('SELECT * FROM stage_runs ORDER BY created_at DESC LIMIT 300')]


@app.get('/api/runs/{identifier}')
def run(identifier: str): return public_run(v.run(identifier))


@app.post('/api/runs/{identifier}/resume')
def resume(identifier: str): return public_run(v.run(v.resume(identifier)))


@app.get('/api/runs/{identifier}/log',response_class=PlainTextResponse)
def log(identifier: str):
    value=v.run(identifier); path=Path(value['log_path'])
    if not path.exists(): return '任务已排队，等待执行器及可用 GPU。'
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-100_000)); content=stream.read().decode('utf-8',errors='replace')
    return content


@app.get('/api/runs/{identifier}/materials')
def materials(identifier: str):
    value=v.run(identifier); directory=v.stage_dir(identifier)
    if value['stage']!='m2': raise ValueError('此阶段没有二维生成材料')
    historical=Path(value['folder'])/'materials'
    if historical.is_dir(): directory=historical
    files=[name for name in v.MATERIALS if (directory/name).is_file()]
    plan=json.loads((directory/'generation_plan.json').read_text()) if (directory/'generation_plan.json').exists() else {}
    return {'files':files,'prompt':plan.get('prompt',''),'api_called':False,'camera':json.loads((directory/'anchor_camera.json').read_text())}


@app.get('/api/runs/{identifier}/bundle')
def bundle(identifier: str): return FileResponse(v.bundle(identifier),filename='qwen_generation_materials.zip')


@app.get('/api/runs/{identifier}/file')
def run_file(identifier: str, name: str, download: bool=False):
    value=v.run(identifier); out=s.unpack(value['outputs'])
    directory=Path(out['directory']) if 'directory' in out else Path(value['folder'])/'outputs'/value['stage']
    historical=Path(value['folder'])/'materials'
    if name in v.MATERIALS and historical.is_dir(): directory=historical
    if name not in set(out.get('files',[]))|set(v.MATERIALS): raise HTTPException(404,'未登记的输出文件')
    path=(directory/name).resolve()
    if not any(path.is_relative_to(root) for root in (s.DATA/'experiments',s.BASE/'outputs')) or not path.is_file():
        raise HTTPException(404,'文件不存在')
    return FileResponse(path,filename=path.name if download else None)


@app.get('/api/assets')
def assets(kind: Optional[str]=None):
    values=s.rows('SELECT * FROM assets ORDER BY created_at DESC')
    if kind=='images': values=[a for a in values if a['kind'] in v.IMAGE_KINDS]
    elif kind=='models': values=[a for a in values if a['kind'] in v.MODEL_KINDS]
    elif kind: values=[a for a in values if a['kind']==kind]
    return [s.public_asset(a) for a in values]


@app.get('/api/assets/{identifier}/file')
def asset_file(identifier: str, download: bool=False):
    value=s.asset(identifier); path=s.asset_path(identifier)
    name=value['name']
    if not name.lower().endswith(path.suffix.lower()): name+=path.suffix
    return FileResponse(path,filename=name if download else None)


@app.get('/api/assets/{identifier}/thumbnail')
def thumbnail(identifier: str):
    value=s.asset(identifier)
    if value['kind'] in v.MODEL_KINDS:
        preview=s.unpack(value['metadata']).get('preview_asset')
        if not preview: raise HTTPException(404,'模型尚无预览图')
        return thumbnail(preview)
    path=s.asset_path(identifier); thumb=s.DATA/'tmp'/(value['sha256']+'.thumb.png')
    if not thumb.exists():
        with Image.open(path) as image:
            image.thumbnail((480,360)); image.save(thumb)
    return FileResponse(thumb)


@app.post('/api/assets/upload')
async def upload(file: UploadFile=File(...), kind: str=Form(...), name: str=Form(''), model: str=Form('external'),
                 prompt: str=Form(''), experiment_id: str=Form(''), camera_aligned: bool=Form(False),
                 transform_json: str=Form('')):
    if kind not in v.IMAGE_KINDS+v.MODEL_KINDS: raise ValueError('只允许上传设备图像、Mesh 或 Gaussian PLY')
    if len(name)>200 or len(prompt)>10000 or len(model)>200: raise ValueError('素材描述过长')
    suffix=Path(file.filename or '').suffix.lower()
    allowed={'.png','.jpg','.jpeg','.webp'} if kind in v.IMAGE_KINDS else {'.glb','.obj','.stl','.ply'}
    if suffix not in allowed: raise ValueError('文件扩展名不支持')
    if kind.startswith('gaussian') and suffix!='.ply': raise ValueError('Gaussian 必须为 PLY')
    meta={'camera_aligned':False}; exp=v.experiment(experiment_id) if experiment_id else None
    if kind in v.IMAGE_KINDS:
        if not exp or not camera_aligned: raise ValueError('请关联实验，并确认图片保留当前 Anchor 完整画布与视角')
        heads=s.unpack(exp['heads']); rid=heads.get('materials')
        if not rid: raise ValueError('请先准备当前实验的生成材料')
        camera=json.loads((v.stage_dir(rid)/'anchor_camera.json').read_text())
        meta.update(camera_aligned=True,camera=camera,camera_key=v.camera_key(camera),scene_sha=exp['scene_sha'],
            alignment_declaration='user confirms full anchor canvas',experiment_id=experiment_id)
    if kind=='gaussian_world':
        if not exp: raise ValueError('世界坐标 PLY 必须指定来源场景实验')
        try:
            transform=json.loads(transform_json or '{}')
            rotation=np.asarray(transform.get('rotation'),dtype=float); translation=np.asarray(transform.get('translation'),dtype=float); scale=float(transform.get('scale',0))
        except (ValueError,TypeError,AttributeError):
            raise ValueError('请输入有效的原始 scale、rotation、translation 变换 JSON') from None
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.isfinite(scale) or scale<=0:
            raise ValueError('请输入有效的原始 scale、rotation、translation 变换 JSON')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):
            raise ValueError('rotation 必须为合法旋转矩阵')
        meta.update(transform=transform,scene_sha=exp['scene_sha'])
    temporary=s.DATA/'tmp'/(s.uid()+suffix); size=0
    try:
        with temporary.open('wb') as stream:
            while chunk:=await file.read(4*1024*1024):
                size+=len(chunk)
                if size>(32*1024**2 if kind in v.IMAGE_KINDS else 4*1024**3): raise ValueError('素材超过上传大小限制')
                stream.write(chunk)
        try: info=s.validate_file(temporary,kind)
        except (OSError,EOFError,KeyError,Image.DecompressionBombError) as exc:
            raise ValueError('无法解析上传文件：'+str(exc)) from None
        meta.update(info)
        if kind in v.IMAGE_KINDS and abs(info['width']/info['height']-camera['width']/camera['height'])>1e-6:
            raise ValueError('图片宽高比与 Anchor 不一致，请保留原始画布')
        previous=s.one('SELECT id FROM assets WHERE sha256=:h AND kind=:k',h=s.sha(temporary),k=kind)
        aid=s.register(temporary,kind,name or file.filename,'external_upload',model,prompt,meta,
            experiment=experiment_id or None,trusted=True)
        result=s.public_asset(s.asset(aid)); result['deduplicated']=bool(previous)
        return result
    finally:
        temporary.unlink(missing_ok=True); await file.close()


@app.get('/api/generations')
def generations():
    result=s.rows('SELECT * FROM generation_records ORDER BY created_at DESC LIMIT 300')
    for row in result:
        for name in ('input_assets','output_assets','metadata'): row[name]=json.loads(row[name])
    return result


@app.get('/')
def index(): return FileResponse(STATIC/'index.html')


app.mount('/static',StaticFiles(directory=STATIC),name='static')
(s.DATA/'m2_batches').mkdir(exist_ok=True)
app.mount('/m2-batches',StaticFiles(directory=s.DATA/'m2_batches',html=True),name='m2-batches')
