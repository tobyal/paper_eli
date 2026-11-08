"""Original-photo SAM selection, real CLIP ranking, and confirmed M1 cases only."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import shutil
import re
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StrictInt
import yaml
from . import store as s
from ..core.context import read_json, write_json
from ..core import m1_reference as ref

router=APIRouter();HOME=s.DATA/'m1_assist'


def init():
    HOME.mkdir(parents=True,exist_ok=True)
    with s.transaction() as conn:
        conn.exec_driver_sql('''CREATE TABLE IF NOT EXISTS m1_jobs (
            id TEXT PRIMARY KEY, selection TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
            folder TEXT NOT NULL, validation INTEGER NOT NULL, request_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, pid INTEGER, error TEXT)''')
        conn.exec_driver_sql('''CREATE TABLE IF NOT EXISTS m1_cases (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES m1_jobs(id),
            name TEXT NOT NULL, folder TEXT NOT NULL, validation INTEGER NOT NULL, created_at TEXT NOT NULL)''')
        conn.exec_driver_sql('''CREATE TABLE IF NOT EXISTS m1_case_archives (
            case_id TEXT PRIMARY KEY REFERENCES m1_cases(id), reason TEXT NOT NULL, archived_at TEXT NOT NULL)''')


def identity(value):
    if not re.fullmatch(r'[a-f0-9]{32,64}',value):raise ValueError('Invalid identifier')
    return value


def job(identifier):
    row=s.one('SELECT * FROM m1_jobs WHERE id=:id',id=identity(identifier))
    if not row:raise HTTPException(404,'M1 job not found')
    return row


def selection_path(identifier):
    path=HOME/'selections'/identity(identifier)
    if not (path/'selection.json').is_file():raise ValueError('请先选择 SAM 区域并获取推荐')
    return path


@lru_cache(maxsize=2)
def text_cache(stamp):
    path=HOME/'vocabulary';metadata=read_json(path/'manifest.json')
    if metadata['library_sha256']!=s.sha(s.BASE/'configs/m1_semantic_library.json'):
        raise ValueError('语义词库已改变，需要先重新编码文本词库（无需训练）')
    with np.load(path/'text_embeddings.npz',allow_pickle=False) as cache:
        return {key:cache[key].copy() for key in cache.files},metadata


def image_response(array):
    buffer=BytesIO();Image.fromarray(array).save(buffer,format='PNG');return Response(buffer.getvalue(),media_type='image/png')


@router.get('/m1-cases')
def page():return FileResponse(Path(__file__).parent/'static/m1_cases.html')


@router.get('/api/m1-assist/photos')
def photos(offset:int=0,limit:int=24,search:str=''):
    if offset<0 or not 1<=limit<=60:raise ValueError('Invalid page')
    items=[]
    for frame,camera in ref.cameras(str(s.ROOT)).items():
        if search.lower() not in frame.lower():continue
        photo,segments,features=ref.frame_paths(s.ROOT,frame)
        if not photo.is_file():continue
        items.append({'name':frame,'cached':segments.is_file() and features.is_file(),
                      'width':camera['width'],'height':camera['height']})
    return {'items':items[offset:offset+limit],'total':len(items),'cached':sum(r['cached'] for r in items),
            'vocabulary_ready':(HOME/'vocabulary/manifest.json').is_file()}


@router.get('/api/m1-assist/photo/{frame}')
def photo(frame:str,thumbnail:bool=False):
    path,_,_=ref.frame_paths(s.ROOT,frame)
    if not thumbnail:return FileResponse(path)
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((300,170));buffer=BytesIO();im.save(buffer,format='JPEG',quality=78)
    return Response(buffer.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=3600'})


class Regions(BaseModel):
    frame:str
    level:StrictInt=Field(ge=1,le=3)
    point:list[StrictInt]|None=None
    box:list[StrictInt]|None=None


class Selection(BaseModel):
    frame:str
    level:StrictInt=Field(ge=1,le=3)
    ids:list[StrictInt]=Field(min_length=1,max_length=64)


@router.post('/api/m1-assist/regions')
def regions(body:Regions):
    if (body.point is None)==(body.box is None):raise ValueError('请点击或框选一种区域')
    if body.point is not None and len(body.point)!=2 or body.box is not None and len(body.box)!=4:raise ValueError('Invalid coordinates')
    segments,_=ref.frame_arrays(s.ROOT,body.frame)
    return {'regions':ref.region_proposals(segments,body.level,body.point,body.box)}


@router.get('/api/m1-assist/region-mask/{frame}/{level}/{index}')
def region_mask(frame:str,level:int,index:int):
    segments,features=ref.frame_arrays(s.ROOT,frame);mask,*_=ref.selected_region(segments,features,level,[index])
    rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=[19,223,191,130];return image_response(rgba)


@router.post('/api/m1-assist/select')
def select(body:Selection):
    segments,features=ref.frame_arrays(s.ROOT,body.frame)
    mask,raw,weights,prototype=ref.selected_region(segments,features,body.level,body.ids)
    cachepath=HOME/'vocabulary/text_embeddings.npz'
    if not cachepath.is_file():raise ValueError('兼容 CLIP 文本词库尚未准备，请稍后重试')
    cache,metadata=text_cache(cachepath.stat().st_mtime_ns)
    rankings=ref.rank_descriptions(prototype,cache['texts'],cache['embeddings'],cache['negatives'])
    photo,sp,fp=ref.frame_paths(s.ROOT,body.frame)
    hashes={'original_photo':s.sha(photo),'sam_cache':s.sha(sp),'clip_cache':s.sha(fp),'vocabulary':s.sha(cachepath)}
    identifier=s.digest({'frame':body.frame,'level':body.level,'ids':sorted(set(body.ids)),'hashes':hashes})
    path=HOME/'selections'/identifier
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.parent/(identifier+'.'+s.uid()+'.tmp');temp.mkdir()
        camera=ref.cameras(str(s.ROOT))[body.frame]
        shutil.copy2(photo,temp/'original_photo.jpg');Image.fromarray(mask.astype(np.uint8)*255).save(temp/'reference_mask.png')
        image=np.asarray(Image.open(photo).convert('RGB').resize((camera['width'],camera['height']))).copy()
        image[mask]=.5*image[mask]+.5*np.array([19,223,191]);Image.fromarray(image).save(temp/'selection_overlay.jpg',quality=95)
        np.save(temp/'clip_region_features.npy',raw);np.save(temp/'clip_prototype.npy',prototype);np.save(temp/'clip_region_weights.npy',weights)
        write_json(temp/'camera.json',camera);write_json(temp/'rankings.json',rankings)
        value={'id':identifier,'frame':body.frame,'level':body.level,'ids':sorted(set(body.ids)),
               'pixels':int(mask.sum()),'camera':camera,'top5':rankings[:5],'hashes':hashes,
               'feature_source':'original photograph cached SAM/CLIP; no Gaussian rendering used for recommendation',
               'pooling':'area-weighted mean of normalized, disjoint same-scale region features; L2 normalize',
               'clip_model':metadata,'created_at':s.now()}
        write_json(temp/'selection.json',value)
        try:temp.rename(path)
        except FileExistsError:shutil.rmtree(temp)
    value=read_json(path/'selection.json')
    return {**{k:value[k] for k in ('id','frame','level','ids','pixels','top5')},'overlay':f'/api/m1-assist/selection/{identifier}/selection_overlay.jpg'}


@router.get('/api/m1-assist/selection/{identifier}/{name}')
def selection_file(identifier:str,name:str):
    if name not in ('reference_mask.png','selection_overlay.jpg','original_photo.jpg','rankings.json','selection.json','clip_prototype.npy'):raise HTTPException(404)
    return FileResponse(selection_path(identifier)/name)


class Localize(BaseModel):
    selection_id:str
    text:str=Field(min_length=3,max_length=250)
    request_id:str=Field(min_length=8,max_length=100)
    validation:bool=False


@router.post('/api/m1-assist/localize')
def localize(body:Localize):
    selection=selection_path(body.selection_id);description=body.text.strip()
    if not any(c.isalpha() for c in description) or not description.isascii():raise ValueError('请选择推荐英文描述，或输入英文描述')
    with s.transaction() as conn:
        existing=s.sql(conn,'SELECT * FROM m1_jobs WHERE request_id=:request',request=body.request_id).mappings().first()
        if existing:
            if existing['selection']!=body.selection_id or existing['text']!=description or bool(existing['validation'])!=body.validation:raise ValueError('Request id reused with different inputs')
            return {'id':existing['id'],'status':existing['status']}
        identifier=s.uid();folder=HOME/('validation_jobs' if body.validation else 'jobs')/identifier;folder.mkdir(parents=True)
        shutil.copytree(selection,folder/'reference')
        config=s.config();config['project'].update(output_dir=str((folder/'outputs').relative_to(s.BASE)),
            records_dir=str((folder/'records').relative_to(s.BASE)),logs_dir=str((folder/'logs').relative_to(s.BASE)))
        config['m1'].update(localization_dir=str((folder/'localization').relative_to(s.ROOT)),target_text=description)
        config['m2'].update(existing_anchor=str((folder/'reference_camera.json').relative_to(s.ROOT)),reuse_anchor=False,
            prompt_mode='generic_device',segmentation_prompt_mode='box_clip')
        config['m3']['input_image']=str((folder/'outputs/m2/object_rgba.png').relative_to(s.BASE))
        (folder/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
        write_json(folder/'reference_camera.json',{'camera':read_json(selection/'camera.json'),'role':'M1 frame only; future M2 must run AVP'})
        write_json(folder/'job.json',{'id':identifier,'root':str(s.ROOT),'selection':str(folder/'reference'),'text':description,
                   'validation':body.validation,'stages':['m1']})
        s.sql(conn,'''INSERT INTO m1_jobs(id,selection,text,status,folder,validation,request_id,created_at)
            VALUES(:id,:selection,:text,'pending',:folder,:validation,:request,:time)''',id=identifier,selection=body.selection_id,
            text=description,folder=str(folder),validation=int(body.validation),request=body.request_id,time=s.now())
    return {'id':identifier,'status':'pending'}


def public_job(row):
    path=Path(row['folder']);result=read_json(path/'result.json') if (path/'success.json').exists() else None
    log=path/'run.log';tail=''
    if log.exists():
        with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-5000));tail=stream.read().decode('utf-8',errors='replace')
    return {**{k:row[k] for k in ('id','selection','text','status','validation','created_at','error')},'result':result,'log':tail}


@router.get('/api/m1-assist/jobs')
def jobs(validation:bool=False):
    return [public_job(r) for r in s.rows('SELECT * FROM m1_jobs WHERE validation=:v ORDER BY created_at DESC LIMIT 30',v=int(validation))]


@router.get('/api/m1-assist/jobs/{identifier}')
def job_status(identifier:str):return public_job(job(identifier))


@router.get('/api/m1-assist/jobs/{identifier}/file/{name:path}')
def job_file(identifier:str,name:str):
    row=job(identifier);folder=Path(row['folder']).resolve();path=(folder/name).resolve()
    if not path.is_relative_to(folder) or name.split('/')[0] not in ('inspection','reference','outputs','localization') or not path.is_file():raise HTTPException(404)
    return FileResponse(path)


class Confirm(BaseModel):
    name:str=Field(min_length=1,max_length=120)


@router.post('/api/m1-assist/jobs/{identifier}/confirm')
def confirm(identifier:str,body:Confirm):
    row=job(identifier)
    if row['status']!='completed':raise ValueError('请等待 M1 定位完成并检查结果后再保存')
    existing=s.one('SELECT * FROM m1_cases WHERE job_id=:id',id=identifier)
    if existing:return existing
    source=Path(row['folder']);integrity=read_json(source/'integrity.json')
    for name,expected in integrity.items():
        if s.sha(source/name)!=expected:raise ValueError('M1 结果文件已改变，不能确认保存：'+name)
    case_id=s.uid();dest=HOME/('validation_cases' if row['validation'] else 'cases')/case_id;temp=dest.with_name(case_id+'.tmp')
    temp.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,temp)
    # A case remains runnable even when the work-in-progress job is archived.
    for path in list(temp.rglob('*.json'))+list(temp.rglob('*.yaml')):
        if path.name=='integrity.json':continue
        text=path.read_text(encoding='utf-8')
        text=text.replace(str(source),str(dest)).replace(str(source.relative_to(s.ROOT)),str(dest.relative_to(s.ROOT))).replace(str(source.relative_to(s.BASE)),str(dest.relative_to(s.BASE)))
        path.write_text(text,encoding='utf-8')
    write_json(temp/'confirmed_case.json',{'id':case_id,'name':body.name,'job_id':identifier,'confirmed_at':s.now(),
        'validation_only':bool(row['validation']),'next_stage':'M2 requires user provided/generated image; nothing automatically launched'})
    receipt_path=temp/'records/m1_status.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path);config=yaml.safe_load((temp/'config.yaml').read_text(encoding='utf-8'))
        for artifact in receipt['artifacts']:
            actual=temp/Path(artifact['path']).relative_to(dest)
            artifact.update(sha256=s.sha(actual),bytes=actual.stat().st_size)
        receipt['config_signature']=s.digest({'stage':config['m1'],'seed':config['project']['seed'],'source_sha256':config['project']['source_sha256']})
        write_json(receipt_path,receipt)
    write_json(temp/'integrity.json',{str(p.relative_to(temp)):s.sha(p) for p in temp.rglob('*') if p.is_file() and p.name!='integrity.json'})
    temp.rename(dest)
    with s.transaction() as conn:
        # Concurrent repeated confirmations of one immutable job are idempotent.
        s.sql(conn,'''INSERT OR IGNORE INTO m1_cases(id,job_id,name,folder,validation,created_at)
            VALUES(:id,:job,:name,:folder,:v,:time)''',id=case_id,job=identifier,name=body.name,folder=str(dest),v=row['validation'],time=s.now())
    saved=s.one('SELECT * FROM m1_cases WHERE job_id=:id',id=identifier)
    if saved['id']!=case_id:shutil.rmtree(dest)
    return saved


@router.get('/api/m1-assist/cases')
def cases(validation:bool=False,include_archived:bool=False):
    return s.rows('''SELECT c.*,j.text,j.selection,a.reason AS archive_reason FROM m1_cases c
        JOIN m1_jobs j ON j.id=c.job_id LEFT JOIN m1_case_archives a ON a.case_id=c.id
        WHERE c.validation=:v AND (:all_cases=1 OR a.case_id IS NULL) ORDER BY c.created_at''',
        v=int(validation),all_cases=int(include_archived))
