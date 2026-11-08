"""Experiment dependency snapshots, stage cache, and output registration."""
import copy
import json
from pathlib import Path
import shutil
import zipfile
import yaml
from . import store as s
from ..core.workflow import current_config,METHOD

IMAGE_KINDS=('edited_scene','object_rgba')
MODEL_KINDS=('mesh','gaussian_local','gaussian_world')
ACTIONS={'m1':('reuse','run'),'m2':('prepare','reuse','qwen'),'m3':('reuse','trellis'),'m4':('refine',)}
MATERIALS=('anchor.png','position_hint.png','edit_mask.png','depth.png','depth_background.png',
           'generation_prompt.txt','anchor_camera.json','conditions.json','generation_plan.json')
SPATIAL_MATERIALS=('generation_region.json','mask_report.json','generation_bbox_mask.png',
    'old_visible_mask.png','old_clean_mask.png','protected_foreground_mask.png','extension_proposal.png')


def experiment(identifier):
    value=s.one('SELECT * FROM experiments WHERE id=:id',id=identifier)
    if not value: raise ValueError('实验不存在')
    return value


def run(identifier):
    value=s.one('SELECT * FROM stage_runs WHERE id=:id',id=identifier)
    if not value: raise ValueError('运行记录不存在')
    return value


def stage_dir(identifier):
    value=run(identifier)
    if value['status']!='completed': raise ValueError('依赖阶段尚未完成')
    return Path(s.unpack(value['outputs'])['directory'])


def active(conn, identifier):
    if s.sql(conn,"SELECT id FROM stage_runs WHERE experiment_id=:e AND status IN ('pending','running')",e=identifier).first():
        raise ValueError('当前实验有排队或运行中的任务，请完成后切换素材')


def create(name, replacement=None, copy_from=None):
    name=name.strip()
    if not 1<=len(name)<=120: raise ValueError('实验名称长度应为 1–120 字符')
    cfg=copy.deepcopy(s.config()); heads={}
    if copy_from:
        parent=experiment(copy_from); cfg=s.unpack(parent['config']); heads=s.unpack(parent['heads'])
    original=cfg
    cfg=current_config(cfg)
    if original['m4'].get('method')!=METHOD:
        heads={k:v for k,v in heads.items() if k not in ('m3','m4')}
    if replacement:
        if len(replacement)>3000: raise ValueError('设备描述过长')
        if replacement!=cfg['m2']['replacement_text']:
            heads={k:v for k,v in heads.items() if k=='m1'}
        cfg['m2']['replacement_text']=replacement
    identifier=s.uid()
    with s.transaction() as conn:
        s.sql(conn,'INSERT INTO experiments VALUES(:id,:name,:scene,:target,:replacement,:cfg,:heads,:t,:t)',
            id=identifier,name=name,scene=cfg['project']['source_sha256'],target=cfg['m1']['target_text'],
            replacement=cfg['m2']['replacement_text'],cfg=s.dump(cfg),heads=s.dump(heads),t=s.now())
        if copy_from:
            for link in s.rows('SELECT * FROM experiment_assets WHERE experiment_id=:e',e=copy_from):
                s.attach(conn,identifier,link['asset_id'],link['role'],bool(link['selected']))
    return identifier


def select(identifier, asset_id, role, bind_camera=False):
    if role not in ('m2','m3'): raise ValueError('只能选择 M2 图像或 M3 模型')
    value=s.asset(asset_id); exp=experiment(identifier)
    if value['kind'] not in (IMAGE_KINDS if role=='m2' else MODEL_KINDS): raise ValueError('素材类型与阶段不匹配')
    meta=s.unpack(value['metadata'])
    if role=='m2':
        if not meta.get('camera_aligned'): raise ValueError('该图像缺少完整参考画布/相机绑定，请从实验页面重新上传并确认画布')
        if meta.get('scene_sha')!=exp['scene_sha']: raise ValueError('图像属于不同原始场景，不能直接用于当前相机监督')
    with s.transaction() as conn:
        active(conn,identifier)
        current=s.sql(conn,'SELECT heads FROM experiments WHERE id=:id',id=identifier).mappings().one()
        heads=s.unpack(current['heads'])
        for key in (('m2','m3','m4') if role=='m2' else ('m3','m4')): heads.pop(key,None)
        s.attach(conn,identifier,asset_id,role,True)
        s.sql(conn,'UPDATE experiments SET heads=:h,updated_at=:t WHERE id=:id',h=s.dump(heads),t=s.now(),id=identifier)


def selected(identifier, role):
    value=s.one('SELECT asset_id FROM experiment_assets WHERE experiment_id=:e AND role=:r AND selected=1',e=identifier,r=role)
    return value['asset_id'] if value else None


def normalized_camera(camera):
    return {**{k:camera[k] for k in ('R','t')},
        **{k:camera[k]/camera['width' if k in ('fx','cx') else 'height'] for k in ('fx','fy','cx','cy')},
        'aspect':camera['width']/camera['height']}


def camera_key(camera): return s.digest(normalized_camera(camera))


def input_fingerprint(key, identifier):
    if key=='asset': return {'sha256':s.asset(identifier)['sha256']}
    value=run(identifier)
    # Identical parameters can produce different generated pixels. Bind the
    # cache to the actual saved artifacts as well as their configuration.
    return {'signature':value['signature'],'artifacts':s.unpack(value['outputs']).get('integrity',{})}


def prerequisites(exp, stage, action):
    heads=s.unpack(exp['heads']); needed=[]
    if stage=='m2': needed=['m1'] if action=='prepare' else ['m1','materials']
    if stage=='m3': needed=['m1','m2']
    if stage=='m4': needed=['m1','m2','m3']
    result={}
    for key in needed:
        if key not in heads: raise ValueError(f'缺少 {key}：请先完成对应阶段')
        stage_dir(heads[key]); result[key]=heads[key]
    if action=='reuse' and stage in ('m2','m3'):
        aid=selected(exp['id'],stage)
        if not aid: raise ValueError('请先选择素材')
        asset=s.asset(aid); s.asset_path(aid)
        result['asset']=aid
        meta=s.unpack(asset['metadata'])
        if stage=='m2':
            cam=json.loads((stage_dir(result['materials'])/'anchor_camera.json').read_text())
            if meta.get('camera_key')!=camera_key(cam): raise ValueError('图像与当前 Anchor 相机不匹配，请上传当前视角图像')
    return result


def submit(identifier, stage, action, request_id, parameters=None, confirm_qwen=False):
    if action not in ACTIONS.get(stage,()): raise ValueError('阶段操作无效')
    if not request_id or len(request_id)>100: raise ValueError('缺少有效幂等请求编号')
    parameters=parameters or {}
    allowed={'m1':set(),'m2':set(),'m3':{'realign','force_model'},'m4':{'iterations'}}
    if set(parameters)-allowed[stage]: raise ValueError('包含此阶段不支持的运行参数')
    if any(type(parameters[k]) is not bool for k in ('realign','force_model') if k in parameters): raise ValueError('开关参数必须为布尔值')
    if action=='qwen' and not confirm_qwen: raise ValueError('必须明确点击并确认一次 Qwen API 调用')
    with s.transaction() as conn:
        old=s.sql(conn,'SELECT * FROM stage_runs WHERE request_id=:r',r=request_id).mappings().first()
        if old:
            if old['experiment_id']!=identifier or old['stage']!=stage or old['action']!=action:
                raise ValueError('幂等请求编号已用于不同任务')
            if s.unpack(old['parameters'])!=parameters: raise ValueError('幂等请求编号对应的参数已改变')
            return old['id']
        active(conn,identifier)
        exp=dict(s.sql(conn,'SELECT * FROM experiments WHERE id=:id',id=identifier).mappings().one())
        inputs=prerequisites(exp,stage,action)
        cfg=current_config(s.unpack(exp['config']))
        if stage=='m1' and cfg['m2'].get('confirmed_m1_case_id'):
            raise ValueError('此实验已绑定人工确认的 M1 案例；请从 M1 案例页选择其他定位，不能换回历史固定设备')
        if action=='qwen' and cfg['m2'].get('preparation_only'):
            raise ValueError('此案例仅准备 Qwen 输入；请下载材料自行生成后上传图像，不提交 API 请求')
        if 'iterations' in parameters:
            steps=parameters['iterations']
            if type(steps) is not int or not 200<=steps<=2000: raise ValueError('M4 局部优化步数必须为 200–2000 的整数')
            cfg['m4']['local_background']['iterations']=steps
        identifier_run=s.uid(); folder=s.DATA/'experiments'/identifier/'runs'/identifier_run
        cfg['project'].update(output_dir=str((folder/'outputs').relative_to(s.BASE)),
            records_dir=str((folder/'records').relative_to(s.BASE)), logs_dir=str((folder/'logs').relative_to(s.BASE)))
        cfg['m3']['input_image']=str((folder/'outputs/m2/object_rgba.png').relative_to(s.BASE))
        key={'version':{'m2':'web-v7-adaptive-visible','m3':'web-v8-pbr-local-gaussian','m4':'web-v8-pbr-scene-integration'}.get(stage,'web-v2'),'stage':stage,'action':action,'scene':exp['scene_sha'],'seed':cfg['project']['seed'],
             'replacement':exp['replacement_text'], 'config':cfg[stage], 'render':cfg['render'],
             'parameters':parameters,'inputs':{k:input_fingerprint(k,v) for k,v in inputs.items()}}
        key['config']=dict(key['config'])
        for name in ('api_key_file','input_image','initialize_from'): key['config'].pop(name,None)
        signature=s.digest(key)
        folder.mkdir(parents=True)
        config_path=folder/'config.yaml'; config_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        if action=='qwen':
            (folder/'authorization.json').write_text(s.dump({'explicit_web_click':True,'request_id':request_id,'at':s.now(),'max_requests':1}),encoding='utf-8')
        s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
            config_path,folder,log_path,created_at,request_id) VALUES(:id,:e,:stage,:action,'pending',:sig,:inputs,:params,:cfg,:folder,:log,:t,:req)''',
            id=identifier_run,e=identifier,stage=stage,action=action,sig=signature,inputs=s.dump(inputs),params=s.dump(parameters),
            cfg=str(config_path),folder=str(folder),log=str(folder/'execution.log'),t=s.now(),req=request_id)
        s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,external_api,status,created_at)
            VALUES(:id,:e,:r,:model,:prompt,:inputs,0,'pending',:t)''',id=s.uid(),e=identifier,r=identifier_run,
            model=(cfg['m2']['model'] if action=='qwen' else cfg['m3']['model'] if action=='trellis' else action),
            prompt=cfg['m2']['replacement_text'],inputs=s.dump(inputs),t=s.now())
    return identifier_run


def resume(identifier):
    value=run(identifier)
    if value['status']!='failed': raise ValueError('仅失败或中断任务可以恢复')
    cfg=yaml.safe_load(Path(value['config_path']).read_text())
    if value['stage'] in ('m3','m4'):
        from ..core.workflow import validate_workflow
        validate_workflow(cfg,stage=value['stage'])
    if value['stage']=='m4' and (Path(value['folder'])/'outputs/m4/background_optimization/training.jsonl').exists() and not (Path(value['folder'])/'outputs/m4/background_optimization/summary.json').exists():
        raise ValueError('局部优化中断记录已保留；请提交新的 M4 运行，不在原目录重新训练')
    with s.transaction() as conn:
        active(conn,value['experiment_id'])
        s.sql(conn,"UPDATE stage_runs SET status='pending',error=NULL,attempt=attempt+1 WHERE id=:id",id=identifier)
        s.sql(conn,"UPDATE generation_records SET status='pending' WHERE run_id=:id",id=identifier)
    return identifier


def output_files(directory):
    allowed={'.png','.jpg','.jpeg','.mp4','.glb','.ply','.json','.txt','.npz','.npy','.pth','.pt','.zip'}
    hidden={'api_request.json','api_response.json','effective_config.json','effective_teacher_config.json'}
    return [str(p.relative_to(directory)).replace('\\','/') for p in sorted(directory.rglob('*'))
        if p.is_file() and p.suffix.lower() in allowed and p.name not in hidden and not any(x.startswith('.') for x in p.relative_to(directory).parts)]


def register_outputs(value, directory):
    inputs=s.unpack(value['inputs']); exp=experiment(value['experiment_id']); stage=value['stage']
    cfg=yaml.safe_load(Path(value['config_path']).read_text()); assets={}
    m2=directory if stage=='m2' else (stage_dir(inputs['m2']) if 'm2' in inputs else None)
    meta={'scene_sha':exp['scene_sha'],'experiment_id':exp['id'],'run_id':value['id']}
    source_asset=s.asset(inputs['asset']) if inputs.get('asset') else None
    if source_asset: meta['source_asset']=source_asset['id']
    plan=m2/'generation_plan.json' if m2 else None
    prompt=json.loads(plan.read_text())['prompt'] if plan and plan.is_file() else cfg['m2']['replacement_text']
    meta['prompt_role']='prepared_generation_prompt'
    if m2 and (m2/'anchor_camera.json').is_file():
        cam=json.loads((m2/'anchor_camera.json').read_text()); meta.update(camera=cam,camera_key=camera_key(cam),camera_aligned=True)
    definitions={
        'edited_scene.png':'edited_scene','object_rgba.png':'object_rgba','object_mask.png':'mask','object_white.png':'preview',
        'depth.png':'depth','depth_background.png':'depth','anchor.png':'preview','position_hint.png':'preview',
        'trellis_asset.glb':'mesh','object_local.ply':'gaussian_local','object_world.ply':'gaussian_world',
        'object_refined_local.ply':'gaussian_local','renders/validation_comparison.jpg':'preview',
        'trellis_native.pt':'checkpoint','trellis_asset_preview.png':'preview',
        'object_fixed.ply':'gaussian_world','object_refined.ply':'gaussian_world',
        'renders/anchor_after.png':'preview',
    }
    if (directory/'transform.json').exists(): meta['transform']=json.loads((directory/'transform.json').read_text())
    if stage=='m4' and 'm3' in inputs:
        transform=stage_dir(inputs['m3'])/'transform.json'
        if transform.exists(): meta['transform']=json.loads(transform.read_text())
        source_assets=s.unpack(run(inputs['m3'])['outputs']).get('assets',{})
        if 'object_refined_local.ply' in source_assets:
            meta['local_asset']=source_assets['object_refined_local.ply']
            if 'trellis_asset.glb' in source_assets: meta['mesh_asset']=source_assets['trellis_asset.glb']
    if m2 and (m2/'object_rgba.png').exists():
        meta['input_image_sha']=s.sha(m2/'object_rgba.png')
        reference=s.one("SELECT id FROM assets WHERE sha256=:h AND kind='object_rgba'",h=meta['input_image_sha'])
        if reference: meta['input_asset']=reference['id']
    model=cfg['m2']['model'] if value['action']=='qwen' or (value['action']=='historical' and stage=='m2') else cfg['m3']['model'] if value['action']=='trellis' or (value['action']=='historical' and stage=='m3') else 'reused / imported'
    if source_asset: model=source_asset['model'] or 'external'
    if stage=='m4': model=('M4 dual-mask background optimization (fixed object)'
        if cfg['m4'].get('method')==METHOD or cfg['m4'].get('background_only')
        else 'Historical DI-SDS (frozen geometry)')
    if stage=='m3' and cfg['m3'].get('method')=='pbr_multiview_gaussian':
        model='PBR multiview RGB+Depth optimized Gaussian / TRELLIS.2 GLB'
    for name,kind in definitions.items():
        file=directory/name
        if file.is_file():
            metadata=dict(meta)
            if kind in IMAGE_KINDS:
                from PIL import Image
                with Image.open(file) as image: metadata.update(width=image.width,height=image.height)
            assets[name]=s.register(file,kind,source=('qwen_api' if value['action']=='qwen' else 'existing_experiment'),
                model=model,prompt=prompt,metadata=metadata,experiment=exp['id'],trusted=True)
    # Keep the local partner/preview on world assets for reuse and display.
    with s.transaction() as conn:
        for name,aid in assets.items():
            record=s.asset(aid); extra=s.unpack(record['metadata'])
            if extra.get('run_id')==value['id']:
                # Refresh descriptive metadata from its own immutable origin;
                # a later deduplicated upload cannot change original provenance.
                extra.update(meta)
                s.sql(conn,'UPDATE assets SET prompt=:p,model=:m WHERE id=:id',p=prompt,m=model,id=aid)
            for key,source in (('local_asset','object_local.ply'),('preview_asset','trellis_asset_preview.png'),('input_asset','object_rgba.png')):
                if source in assets: extra[key]=assets[source]
            if 'object_refined_local.ply' in assets: extra['local_asset']=assets['object_refined_local.ply']
            if 'renders/validation_comparison.jpg' in assets: extra['preview_asset']=assets['renders/validation_comparison.jpg']
            if record['kind'] in MODEL_KINDS:
                for key,source in (('mesh_asset','trellis_asset.glb'),('world_asset','object_world.ply'),('checkpoint_asset','trellis_native.pt')):
                    if source in assets: extra[key]=assets[source]
            if 'preview_asset' not in extra and 'renders/anchor_after.png' in assets: extra['preview_asset']=assets['renders/anchor_after.png']
            s.sql(conn,'UPDATE assets SET metadata=:m WHERE id=:id',m=s.dump(extra),id=aid)
    return assets


def finish(identifier, directory, reused_from=None, seconds=0):
    value=run(identifier); directory=Path(directory)
    assets=register_outputs(value,directory)
    files=output_files(directory)
    # Hash all exposed files, so cache reuse checks actual bytes, not just an old status flag.
    integrity={name:{'bytes':(directory/name).stat().st_size,'sha256':s.sha(directory/name)} for name in files}
    outputs={'directory':str(directory),'files':files,'assets':assets,'integrity':integrity}
    with s.transaction() as conn:
        s.sql(conn,"UPDATE stage_runs SET status='completed',outputs=:o,reused_from=:reuse,finished_at=:t,error=NULL WHERE id=:id",
            o=s.dump(outputs),reuse=reused_from,t=s.now(),id=identifier)
        exp=dict(s.sql(conn,'SELECT * FROM experiments WHERE id=:id',id=value['experiment_id']).mappings().one())
        heads=s.unpack(exp['heads']); key='materials' if value['action']=='prepare' else value['stage']
        invalid={'m1':['materials','m2','m3','m4'],'materials':['m2','m3','m4'],'m2':['m3','m4'],'m3':['m4'],'m4':[]}
        for downstream in invalid[key]: heads.pop(downstream,None)
        heads[key]=identifier
        s.sql(conn,'UPDATE experiments SET heads=:h,updated_at=:t WHERE id=:id',h=s.dump(heads),t=s.now(),id=exp['id'])
        generated_role='m2' if value['action']=='qwen' else 'm3' if value['action']=='trellis' else None
        generated_asset=assets.get('edited_scene.png' if generated_role=='m2' else 'trellis_asset.glb')
        if generated_role and generated_asset: s.attach(conn,exp['id'],generated_asset,generated_role,True)
        lock=Path(value['folder'])/'records/qwen_generation_once.lock'
        prompt_file=directory/'generation_plan.json'
        if not prompt_file.is_file() and 'm2' in s.unpack(value['inputs']):
            prompt_file=stage_dir(s.unpack(value['inputs'])['m2'])/'generation_plan.json'
        prompt=json.loads(prompt_file.read_text())['prompt'] if prompt_file.exists() else exp['replacement_text']
        metadata={'stage_cache_hit':bool(reused_from),'trellis_invoked':value['stage']=='m3' and (directory/'trellis_generation.json').is_file() and not reused_from}
        s.sql(conn,"UPDATE generation_records SET status='completed',seconds=:sec,external_api=:api,output_assets=:out,prompt=:prompt,metadata=:meta WHERE run_id=:id",
            sec=seconds,api=int(lock.exists()),out=s.dump(assets),prompt=prompt,meta=s.dump(metadata),id=identifier)


def verify_outputs(value):
    out=s.unpack(value['outputs']); base=Path(out['directory'])
    return bool(out.get('integrity')) and all((base/name).is_file() and (base/name).stat().st_size==v['bytes']
        and s.sha(base/name)==v['sha256'] for name,v in out['integrity'].items())


def bundle(run_id):
    value=run(run_id)
    if value['stage']!='m2' or value['status']!='completed': raise ValueError('请先准备 M2 材料')
    directory=stage_dir(run_id); target=Path(value['folder'])/'generation_materials.zip'
    if value['action']=='historical' and (Path(value['folder'])/'materials').is_dir():
        directory=Path(value['folder'])/'materials'
    missing=[name for name in MATERIALS if not (directory/name).is_file()]
    if missing: raise ValueError('材料缺失：'+', '.join(missing))
    if not target.is_file():
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for name in MATERIALS: z.write(directory/name,name)
            for name in ('depth.npz','generation_inputs.json',*SPATIAL_MATERIALS):
                if (directory/name).is_file():z.write(directory/name,name)
    return target


def bootstrap():
    """Register the completed historical run read-only; never rerun its algorithms."""
    s.init()
    if s.one('SELECT id FROM experiments LIMIT 1'): return
    eid=create('原始基线 · 黄色机械臂')
    heads={}; cfg=s.config(); baseline=s.BASE/'outputs'
    if (baseline/'m4/object_refined.ply').exists() and not (baseline/'m4/object_fixed.ply').exists():
        # Keep an imported historical result distinct from the current method.
        cfg['m4']={'method':'di_sds','iterations':2000}
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET config=:cfg WHERE id=:id',cfg=s.dump(cfg),id=eid)
    for stage in ('m1','m2','m3','m4'):
        directory=baseline/stage; receipt=s.BASE/'records'/f'{stage}_status.json'
        if not receipt.is_file() or not directory.is_dir(): continue
        old=json.loads(receipt.read_text())
        if old.get('status')!='complete': continue
        if not all(Path(a['path']).is_file() and s.sha(a['path'])==a['sha256'] for a in old['artifacts']):
            raise ValueError('历史产物校验失败：'+stage)
        rid=s.uid(); folder=s.DATA/'experiments'/eid/'runs'/rid; folder.mkdir(parents=True)
        cp=folder/'config.yaml'; cp.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        inputs={k:v for k,v in heads.items() if k in ('m1','m2','m3')}
        with s.transaction() as conn:
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,config_path,folder,log_path,created_at,request_id)
                VALUES(:id,:e,:stage,'historical','running',:sig,:inputs,'{}',:cfg,:folder,:log,:t,:id)''',
                id=rid,e=eid,stage=stage,sig=s.digest({'baseline_receipt':old}),inputs=s.dump(inputs),cfg=str(cp),folder=str(folder),
                log=str(s.BASE/'logs'/f'{stage}.log'),t=s.now())
            s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,status,created_at)
                VALUES(:id,:e,:r,:model,:prompt,:inputs,'completed',:t)''',id=s.uid(),e=eid,r=rid,model='historical baseline',
                prompt=cfg['m2']['replacement_text'],inputs=s.dump(inputs),t=s.now())
        finish(rid,directory)
        heads[stage]=rid
        if stage=='m2':
            # The historical prompt is reused verbatim; this file is an independent material export.
            materials=folder/'materials'; materials.mkdir()
            for name in (*MATERIALS,*SPATIAL_MATERIALS):
                if (directory/name).exists(): shutil.copy2(directory/name,materials/name)
            plan=json.loads((directory/'generation_plan.json').read_text())
            (materials/'generation_prompt.txt').write_text(plan['prompt'],encoding='utf-8')
            heads['materials']=rid
    with s.transaction() as conn:
        s.sql(conn,'UPDATE experiments SET heads=:h WHERE id=:id',h=s.dump(heads),id=eid)
        for role,rid,name in [('m2',heads.get('m2'),'object_rgba.png'),('m3',heads.get('m3'),'trellis_asset.glb')]:
            if rid:
                aid=s.unpack(run(rid)['outputs'])['assets'].get(name)
                if aid: s.attach(conn,eid,aid,role,True)
