"""Confirmed M1 cases -> existing M2 prepare queue, with immutable versions."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import store as s, service as v, m1_cases as m1
from ..core.context import read_json, write_json
from ..core.m2_prompt import build_prompt
from ..core.workflow import current_config

router=APIRouter()


def init():
    with s.transaction() as conn:
        conn.exec_driver_sql('''CREATE TABLE IF NOT EXISTS case_m2_preparations (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES m1_cases(id),
            experiment_id TEXT NOT NULL REFERENCES experiments(id), run_id TEXT,
            replacement TEXT NOT NULL, prompt_override TEXT, validation INTEGER NOT NULL,
            request_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)''')


def case(identifier):
    row=s.one('SELECT * FROM m1_cases WHERE id=:id',id=m1.identity(identifier))
    if not row:raise HTTPException(404,'请先确认保存一个 M1 案例')
    folder=Path(row['folder']).resolve();folder.relative_to(m1.HOME.resolve())
    if read_json(folder/'confirmed_case.json')['id']!=row['id']:raise ValueError('M1 确认记录不匹配')
    return row,folder


class PromptInput(BaseModel):
    replacement_text:str=Field(min_length=3,max_length=3000)


class Preparation(PromptInput):
    prompt_override:str|None=Field(default=None,min_length=3,max_length=12000)
    request_id:str=Field(min_length=8,max_length=100)
    avp_review_fallback:bool=False


def config_for_case(row,folder,description,override=None):
    description=description.strip()
    if not description.isascii() or not any(c.isalpha() for c in description):
        raise ValueError('请填写新设备的英文描述')
    cfg=current_config(yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8')))
    cfg['m1'].update(localization_dir=str((folder/'localization').relative_to(s.ROOT)),
        target_text=read_json(folder/'localization/target.json')['text'])
    cfg['m2'].update(replacement_text=description,prompt_mode='generic_device',reuse_anchor=False,
        avp_observed_fallback=True,confirmed_m1_case_id=row['id'],preparation_only=True,
        segmentation_prompt_mode='box_clip')
    cfg['m2'].pop('prompt_override',None)
    if override is not None:
        if not override.strip():raise ValueError('Prompt 不能为空')
        cfg['m2']['prompt_override']=override.strip()
    return cfg


@router.get('/m2-cases')
def page():return FileResponse(Path(__file__).parent/'static/m2_cases.html')


@router.get('/api/m1-assist/cases/{identifier}/m2/description-options')
def description_options(identifier:str):
    case(identifier)
    catalog=read_json(s.BASE/'configs/m2_case_description_options.json')
    entry=next((item for item in catalog['cases'] if item['case_id']==identifier),None)
    return entry or {'case_id':identifier,'options':[],
                     'design_note':'此案例尚无预写方案，可以直接填写新设备英文描述。'}


@router.post('/api/m1-assist/cases/{identifier}/m2/prompt')
def prompt(identifier:str,body:PromptInput):
    row,folder=case(identifier);cfg=config_for_case(row,folder,body.replacement_text)
    return {'prompt':build_prompt(SimpleNamespace(cfg=cfg)),'target_text':cfg['m1']['target_text'],
            'image_order':['anchor.png','position_hint.png','depth_background.png']}


def submit_existing(preparation):
    if not preparation['run_id']:
        rid=v.submit(preparation['experiment_id'],'m2','prepare','case-m2-'+preparation['id'])
        with s.transaction() as conn:s.sql(conn,'UPDATE case_m2_preparations SET run_id=:run WHERE id=:id',run=rid,id=preparation['id'])
    return detail(preparation['id'])


@router.post('/api/m1-assist/cases/{identifier}/m2/prepare')
def prepare(identifier:str,body:Preparation):
    if s.one('SELECT case_id FROM m1_case_archives WHERE case_id=:id',id=identifier):
        raise ValueError('此 M1 案例已归档，不再创建新的 M2 输入')
    row,folder=case(identifier);cfg=config_for_case(row,folder,body.replacement_text,body.prompt_override)
    if body.avp_review_fallback:
        cfg['m2'].update(avp_review_fallback=True,avp_review_policy='expanded_observed_canvas_v1')
    existing=s.one('SELECT * FROM case_m2_preparations WHERE request_id=:r',r=body.request_id)
    if existing:
        stored=s.unpack(v.experiment(existing['experiment_id'])['config'])
        if bool(stored['m2'].get('avp_review_fallback'))!=body.avp_review_fallback:
            raise ValueError('同一请求编号不能改变 Anchor 人工检查策略')
        if (existing['case_id'],existing['replacement'],existing['prompt_override'])!=(identifier,cfg['m2']['replacement_text'],cfg['m2'].get('prompt_override')):
            raise ValueError('同一请求编号不能用于不同案例、描述或 Prompt')
        return submit_existing(existing)
    integrity=read_json(folder/'integrity.json')
    required=['config.yaml','localization/target.json',*[f'outputs/m1/{n}' for n in
        ('manifest.json','edit_region.json','target.json','target_gaussians.ply','background_gaussians.ply','target_indices.npy','keep_indices.npy')]]
    for name in required:
        if name not in integrity or s.sha(folder/name)!=integrity[name]:raise ValueError('M1 案例产物校验失败：'+name)
    eid=s.uid();rid=s.uid();pid=s.uid();work=s.DATA/'experiments'/eid/'runs'/rid
    dest=work/'outputs/m1';dest.parent.mkdir(parents=True)
    shutil.copytree(folder/'outputs/m1',dest)
    manifest=read_json(dest/'manifest.json')
    for key in ('target_gaussians','background_gaussians'):manifest[key]=str(dest/(key+'.ply'))
    manifest.update(confirmed_case_id=identifier,m1_executed=False,confirmed_case_directory=str(folder))
    write_json(dest/'manifest.json',manifest)
    files=v.output_files(dest)
    outputs={'directory':str(dest),'files':files,'assets':{},
        'integrity':{n:{'bytes':(dest/n).stat().st_size,'sha256':s.sha(dest/n)} for n in files}}
    cfg['project'].update(output_dir=str((work/'outputs').relative_to(s.BASE)),
        records_dir=str((work/'records').relative_to(s.BASE)),logs_dir=str((work/'logs').relative_to(s.BASE)))
    cp=work/'config.yaml';cp.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (work/'execution.log').write_text('Read verified confirmed M1 outputs. M1 was not rerun.\n',encoding='utf-8')
    write_json(work/'case_source.json',{'case_id':identifier,'folder':str(folder),'input_hashes':{n:integrity[n] for n in required},'m1_executed':False})
    with s.transaction() as conn:
        # Serialized unique request check also covers concurrent double clicks.
        old=s.sql(conn,'SELECT * FROM case_m2_preparations WHERE request_id=:r',r=body.request_id).mappings().first()
        if old:
            # Newly allocated files are outside any earlier experiment.
            shutil.rmtree(s.DATA/'experiments'/eid)
            stored=s.unpack(v.experiment(old['experiment_id'])['config'])
            if bool(stored['m2'].get('avp_review_fallback'))!=body.avp_review_fallback:
                raise ValueError('同一请求编号不能改变 Anchor 人工检查策略')
            if (old['case_id'],old['replacement'],old['prompt_override'])!=(identifier,cfg['m2']['replacement_text'],cfg['m2'].get('prompt_override')):
                raise ValueError('请求编号的参数已改变')
            existing=dict(old)
        else:
            s.sql(conn,'''INSERT INTO experiments VALUES(:id,:name,:scene,:target,:replacement,:cfg,:heads,:t,:t)''',
                id=eid,name=(('验证 · ' if row['validation'] else '')+'M2 输入 · '+row['name'])[:120],scene=cfg['project']['source_sha256'],
                target=cfg['m1']['target_text'],replacement=cfg['m2']['replacement_text'],cfg=s.dump(cfg),heads=s.dump({'m1':rid}),t=s.now())
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,config_path,folder,outputs,log_path,created_at,finished_at,request_id)
                VALUES(:id,:e,'m1','case_import','completed',:sig,'{}','{}',:cfg,:folder,:outputs,:log,:t,:t,:id)''',
                id=rid,e=eid,sig=s.digest({'case_id':identifier,'integrity':integrity}),cfg=str(cp),folder=str(work),outputs=s.dump(outputs),log=str(work/'execution.log'),t=s.now())
            s.sql(conn,'''INSERT INTO case_m2_preparations(id,case_id,experiment_id,replacement,prompt_override,validation,request_id,created_at)
                VALUES(:id,:case,:e,:replacement,:prompt,:validation,:request,:t)''',id=pid,case=identifier,e=eid,
                replacement=cfg['m2']['replacement_text'],prompt=cfg['m2'].get('prompt_override'),validation=row['validation'],request=body.request_id,t=s.now())
            existing={'id':pid,'experiment_id':eid,'run_id':None}
    return submit_existing(existing)


@router.get('/api/m1-assist/cases/{identifier}/m2')
def preparations(identifier:str):
    case(identifier)
    return [detail(r['id']) for r in s.rows('SELECT id FROM case_m2_preparations WHERE case_id=:c ORDER BY created_at DESC',c=identifier)]


@router.get('/api/m2-cases/{identifier}')
def detail(identifier:str):
    row=s.one('SELECT * FROM case_m2_preparations WHERE id=:id',id=m1.identity(identifier))
    if not row:raise HTTPException(404,'M2 准备记录不存在')
    result=dict(row);result['status']='submission_pending';result['error']=None;result['log']=''
    if row['run_id']:
        run=v.run(row['run_id']);result.update(status=run['status'],error=run['error'])
        path=Path(run['log_path'])
        if path.exists():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-6000));result['log']=f.read().decode('utf-8',errors='replace')
        if run['status']=='completed':
            directory=v.stage_dir(run['id'])
            result.update(prompt=(directory/'generation_prompt.txt').read_text(encoding='utf-8'),
                conditions=read_json(directory/'conditions.json'),input_manifest=read_json(directory/'generation_inputs.json'),
                output_directory=str(directory),api_called=False)
    result['experiment_url']='/#experiment/'+row['experiment_id']
    return result
