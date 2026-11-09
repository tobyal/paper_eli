"""Prepare new, wider M2 views without generation or changing completed cases."""
import argparse
import copy
import html
import math
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import yaml
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.context import Context, gsle, read_json, write_json, sha256
from industrial_swap_v2.core.m2_space import build_generation_region, settings
from industrial_swap_v2.core.scene import read_vertices

DESCRIPTIONS = {
    'case_04_B': 'A compact industrial process chiller with a light-gray and white rectangular enclosure, a circular dark-gray ventilation grille, a small blue upper control panel, and four caster wheels under its integrated gray base. Use predominantly neutral light-gray and white painted metal with restrained blue accents. Keep the unit compact, with no external pipes or protruding parts.',
    'case_07_A': 'A wide automated parts-storage cabinet with a white and light-gray rectangular enclosure, blue horizontal drawer fronts, one central dispensing hatch, and a small integrated touchscreen. All drawers are closed. The cabinet stands on a solid gray plinth. Use only white, gray and muted industrial blue for its painted body panels.',
    'case_10_C': 'A slim industrial tool cabinet with a light-gray rectangular body, muted blue vertically stacked shallow drawer fronts, a gray flat top, recessed gray handles, and four small caster wheels integrated underneath. All drawers remain closed. Use only light-gray, white and muted blue for its painted metal body.',
}
INPUTS = {'01_scene_rgb.png':'anchor.png', '02_position_hint.png':'position_hint.png',
          '03_background_depth.png':'depth_background.png', 'prompt.txt':'generation_prompt.txt'}


def candidates(batch, gpu):
    if batch.exists(): raise FileExistsError('Use a new batch directory')
    batch.mkdir(parents=True)
    root=Path.cwd(); items=[]
    for key, description in DESCRIPTIONS.items():
        source=Context(root/'industrial_swap_v2/outputs/registration_all_cases_v1'/key/'config.yaml')
        out=batch/key;out.mkdir();(out/'m1').symlink_to(source.stage('m1').resolve(),target_is_directory=True)
        cfg=copy.deepcopy(source.cfg);relative=str(out.relative_to(source.base))
        cfg['project'].update(output_dir=relative,records_dir=relative+'/records',logs_dir=relative+'/logs',gpu=gpu)
        cfg['m2'].update(replacement_text=description,reuse_anchor=True,
            existing_anchor=str(out/'selected_anchor.json'),preparation_only=True,prompt_mode='generic_device')
        cfg['m2'].pop('prompt_override',None)
        (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        ctx=Context(out/'config.yaml');c=gsle(ctx)
        import avp
        region=build_generation_region(read_vertices(ctx.stage('m1')/'target_gaussians.ply'),
            read_json(ctx.stage('m1')/'target.json'),description,settings(cfg))
        old=c.Camera.from_record(read_json(source.stage('m2')/'anchor_camera.json'))
        center=np.array(region['center']);up=np.array(region['up']);corners=np.array(region['bbox_corners'])
        direction=old.center-center;distance=float(np.linalg.norm(direction));pool=[]
        for scale in (1.25,1.5):
            for angle in (-35,-20,0,20,35):
                position=center+scale*Rotation.from_rotvec(up*np.deg2rad(angle)).apply(direction)
                name=f'wider_{scale:.2f}_yaw_{angle:+03d}'
                cam=c.look_at(position,center,up,512,49.1,name)
                pool.append((cam,{'pool':'wider_orbit','distance_ratio':scale,'yaw_offset':angle}))
        observed=[]
        for cam in c.source_cameras(938):
            vector=cam.center-center;ratio=float(np.linalg.norm(vector)/distance)
            angle=float(np.rad2deg(np.arccos(np.clip(vector@direction/np.linalg.norm(vector)/distance,-1,1))))
            if ratio<1.12 or angle>75:continue
            xy,z=cam.project(corners)
            if np.min(z)<=.01:continue
            lo,hi=xy.min(0),xy.max(0);span=float(max(hi-lo))
            side=int(math.ceil(max(span*2.2,cam.height*.55)))
            if side>cam.height or side<64:continue
            midpoint=(lo+hi)/2
            left=int(np.clip(round(midpoint[0]-side/2),0,cam.width-side))
            top=int(np.clip(round(midpoint[1]-side/2),0,cam.height-side))
            if not (lo>np.array([left,top])+5).all() or not (hi<np.array([left+side,top+side])-5).all():continue
            f=512/side; cropped=c.Camera(cam.R,cam.t,512,512,cam.fx*f,cam.fy*f,
                (cam.cx-left)*f,(cam.cy-top)*f,'observed_'+Path(cam.name).stem)
            observed.append((angle+4*abs(ratio-1.5),cropped,{'pool':'calibrated_observed',
                'distance_ratio':ratio,'source_camera':cam.name,'crop_xyxy':[left,top,left+side,top+side]}))
        chosen_centers=[]
        for _,cam,meta in sorted(observed,key=lambda x:x[0]):
            if any(np.linalg.norm(cam.center-p)<distance*.08 for p in chosen_centers):continue
            pool.append((cam,meta));chosen_centers.append(cam.center)
            if len(chosen_centers)==6:break
        scene=c.GaussianData.load(ctx.source(cfg['project']['source_ply']))
        ids=np.load(ctx.stage('m1')/'target_indices.npy');obj=scene.subset(ids)
        flags=torch.zeros(len(scene.xyz),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
        folder=out/'candidates';folder.mkdir();records=[]
        with torch.no_grad():
            for index,(cam,meta) in enumerate(pool):
                joined=c.render(cam,scene,flags);isolated=c.render(cam,obj)
                mask=(isolated['alpha'][0]>.5).cpu().numpy()
                visible=float(joined['object_alpha'].sum()/isolated['alpha'].sum().clamp_min(1e-6))
                coverage=float((joined['alpha']>.5).float().mean())
                xy,z=cam.project(corners);margin=float(np.min(np.r_[xy.min(0),512-xy.max(0)])/512)
                score=avp.brightness_scores(isolated['rgb'].permute(1,2,0).cpu().numpy(),mask)
                rec={'index':index,'camera':cam.record(),'visibility':visible,'scene_coverage':coverage,
                    'generation_margin':margin,'target_pixels_fraction':float(mask.mean()),'score':score,
                    'eligible':bool(visible>=.9 and coverage>=.95 and margin>=.04 and mask.mean()>.015),**meta}
                records.append(rec);c.save_image(folder/f'{index:02d}.png',joined['rgb'])
                c.save_image(folder/f'{index:02d}_mask.png',joined['object_alpha'])
                print(key,index,cam.name,'visibility',round(visible,3),'coverage',round(coverage,3),flush=True)
        board=Image.new('RGB',(400*4,440*math.ceil(len(records)/4)),'white');draw=ImageDraw.Draw(board)
        for r in records:
            i=r['index'];x=i%4*400;y=i//4*440
            board.paste(Image.open(folder/f'{i:02d}.png').resize((400,400)),(x,y+40))
            draw.text((x+5,y+4),f"{i:02d} {r['camera']['name']}",fill='black')
            draw.text((x+5,y+20),f"V={r['visibility']:.2f} cover={r['scene_coverage']:.2f} dist={r['distance_ratio']:.2f}x",fill='black')
        board.save(out/'candidate_views.jpg',quality=92)
        write_json(out/'candidates.json',records)
        items.append({'key':key,'description':description,'folder':str(out),
            'source_config':str(source.config_path),'previous_anchor':old.record(),'original_distance':distance})
        write_json(batch/'manifest.json',{'items':items,'qwen_called':False,'training_executed':False})
        del scene,obj,flags,joined,isolated;torch.cuda.empty_cache()


def extra_views(batch,key,level_only=False):
    """Small elevation sweep when pulling back introduces foreground occlusion."""
    item=next(i for i in read_json(batch/'manifest.json')['items'] if i['key']==key)
    out=Path(item['folder']);ctx=Context(out/'config.yaml');c=gsle(ctx)
    import avp
    region=build_generation_region(read_vertices(ctx.stage('m1')/'target_gaussians.ply'),
        read_json(ctx.stage('m1')/'target.json'),item['description'],settings(ctx.cfg))
    old=c.Camera.from_record(item['previous_anchor']);center=np.asarray(region['center']);up=np.asarray(region['up'])
    direction=old.center-center;distance=np.linalg.norm(direction);elevation=np.arcsin(direction@up/distance)
    horizontal=direction-up*(direction@up);horizontal/=np.linalg.norm(horizontal)
    scene=c.GaussianData.load(ctx.source(ctx.cfg['project']['source_ply']))
    ids=np.load(ctx.stage('m1')/'target_indices.npy');obj=scene.subset(ids)
    flags=torch.zeros(len(scene.xyz),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
    records=read_json(out/'candidates.json');new=[]
    with torch.no_grad():
        for scale in ((1.15,1.1) if level_only else (1.15,1.25)):
            for delta in ((0,) if level_only else (-10,-5,5)):
                e=elevation+np.deg2rad(delta);position=center+distance*scale*(np.cos(e)*horizontal+np.sin(e)*up)
                cam=c.look_at(position,center,up,512,49.1,f'wider_{scale:.2f}_elev_{delta:+03d}')
                joined=c.render(cam,scene,flags);isolated=c.render(cam,obj)
                mask=(isolated['alpha'][0]>.5).cpu().numpy();visible=float(joined['object_alpha'].sum()/isolated['alpha'].sum().clamp_min(1e-6))
                coverage=float((joined['alpha']>.5).float().mean());xy,z=cam.project(np.asarray(region['bbox_corners']))
                margin=float(np.min(np.r_[xy.min(0),512-xy.max(0)])/512)
                record={'index':len(records),'camera':cam.record(),'visibility':visible,'scene_coverage':coverage,
                    'generation_margin':margin,'target_pixels_fraction':float(mask.mean()),
                    'score':avp.brightness_scores(isolated['rgb'].permute(1,2,0).cpu().numpy(),mask),
                    'eligible':bool(visible>=.9 and coverage>=.95 and margin>=.04 and mask.mean()>.015),
                    'pool':'wider_elevation_sweep','distance_ratio':scale,'elevation_delta':delta}
                records.append(record);new.append(record);c.save_image(out/'candidates'/f'{record["index"]:02d}.png',joined['rgb'])
                print(key,record['index'],cam.name,'visibility',visible,'eligible',record['eligible'],flush=True)
    write_json(out/'candidates.json',records)
    board=Image.new('RGB',(1200,880),'white');draw=ImageDraw.Draw(board)
    for j,r in enumerate(new):
        x=j%3*400;y=j//3*440;board.paste(Image.open(out/'candidates'/f'{r["index"]:02d}.png').resize((400,400)),(x,y+40))
        draw.text((x+6,y+8),f'{r["index"]} {r["camera"]["name"]} V={r["visibility"]:.3f}',fill='black')
    board.save(out/'extra_views.jpg',quality=93)


def prepare(batch,selections):
    from industrial_swap_v2.modules.m2_image_generation import run
    manifest=read_json(batch/'manifest.json')
    for item in manifest['items']:
        out=Path(item['folder']);ctx=Context(out/'config.yaml')
        record=read_json(out/'candidates.json')[selections[item['key']]]
        if record['scene_coverage']<.95 or record['generation_margin']<.04:
            raise ValueError('Selected view failed scene coverage/framing checks')
        if not record['eligible'] and not ctx.cfg['m2'].get('avp_review_fallback'):
            raise ValueError('Below-threshold view requires the existing input-only review fallback')
        write_json(out/'selected_anchor.json',{'camera':record['camera'],'selected':record,
            'quality_status':'passed' if record['eligible'] else 'manual_review_required',
            'quality_note':None if record['eligible'] else 'Input-only review: target contribution is below the unchanged 90% AVP threshold; not marked as AVP-eligible.',
            'scoring':'visibility/framing review + existing AVP brightness diagnostics',
            'candidate_pool':record['pool'],'adaptation':'User requested a new, more distant Anchor. Original M1 and completed M2 remain unchanged.',
            'thresholds':{'min_visibility':.9,'min_scene_coverage':.95,'min_margin':.04}})
        run(ctx,prepare_only=True)
        m2=ctx.stage('m2');mask=np.array(Image.open(m2/'edit_mask.png'))>0
        before=np.array(Image.open(m2/'depth.png'));after=np.array(Image.open(m2/'depth_background.png'))
        assert np.array_equal(before[~mask],after[~mask]) and np.all(after[mask]==0)
        assert not list(ctx.records.glob('qwen*lock'))
        item.update(selected=record,mask_report=read_json(m2/'mask_report.json'),status='prepared')
        write_json(batch/'manifest.json',manifest)


def export(batch):
    manifest=read_json(batch/'manifest.json');assert all(i.get('status')=='prepared' for i in manifest['items'])
    root=Path.cwd();web=root/'industrial_swap_v2/webapp/static/batches'/batch.name
    web.mkdir(parents=True,exist_ok=True);cards=[];files={}
    board=Image.new('RGB',(1536,3*552),'white');draw=ImageDraw.Draw(board)
    for row,item in enumerate(manifest['items']):
        key=item['key'];short=key.replace('case_','case')[:6];out=Path(item['folder']);dest=web/short;dest.mkdir(exist_ok=True)
        qwen=batch/'qwen_only'/short;qwen.mkdir(parents=True,exist_ok=True)
        for name,source in INPUTS.items():
            shutil.copy2(out/'m2'/source,qwen/name);shutil.copy2(qwen/name,dest/name)
            files[f'{short}/{name}']=sha256(qwen/name)
        source=Context(item['source_config']);shutil.copy2(source.stage('m2')/'anchor.png',dest/'previous_anchor.png')
        for column,name in enumerate(('01_scene_rgb.png','02_position_hint.png','03_background_depth.png')):
            board.paste(Image.open(dest/name).convert('RGB').resize((512,512)),(column*512,row*552+40))
            draw.text((column*512+8,row*552+12),f'{short} | '+name,fill='black')
        with zipfile.ZipFile(web/(short+'_qwen_inputs.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for name in INPUTS:z.write(qwen/name,name)
        camera=item['selected'];prompt=(qwen/'prompt.txt').read_text()
        if not camera['eligible']:
            cards.append(f'<p>{short}：旧目标可见贡献 {camera["visibility"]:.1%}，略低于原 90% 门槛；此输入保留人工检查标记，未降低门槛。</p>')
        cards.append(f'<section><h2>{short}</h2><p>{html.escape(item["description"])}</p><p>新视角 {html.escape(camera["camera"]["name"])}；相机距离 {camera["distance_ratio"]:.2f}×；旧设备可见贡献 {camera["visibility"]:.1%}。</p><p><a href="{short}_qwen_inputs.zip">下载本例 Qwen 输入</a></p><div class="images">'+''.join(f'<figure><img src="{short}/{n}"><figcaption>{t}</figcaption></figure>' for n,t in [('previous_anchor.png','旧 Anchor（仅供对比）'),('01_scene_rgb.png','新 Anchor / 图1'),('02_position_hint.png','位置提示 / 图2'),('03_background_depth.png','背景深度 / 图3')])+f'</div><details><summary>Qwen Prompt</summary><pre>{html.escape(prompt)}</pre><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent)">复制 Prompt</button></details></section>')
    archive=batch/'qwen_inputs_case04_case07_case10.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(batch/'qwen_only'/name,name)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==12 and z.testzip() is None
    shutil.copy2(archive,web/archive.name);board.save(batch/'new_qwen_inputs_overview.jpg',quality=93)
    shutil.copy2(batch/'new_qwen_inputs_overview.jpg',web/'overview.jpg')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新视角 Qwen 输入：04 / 07 / 10</title><style>body{font:16px/1.7 system-ui;background:#eef2f6;max-width:1700px;margin:auto;padding:24px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%}pre{white-space:pre-wrap}a{color:#1659ad}@media(max-width:900px){.images{grid-template-columns:1fr 1fr}}</style><h1>case04 / case07 / case10：新视角 Qwen 输入</h1><p>每例仅提交三张图和 Prompt。保持图1→图2→图3顺序。未调用 Qwen；原案例结果保留。</p><a href="'+archive.name+'">下载全部输入 ZIP（仅12个输入文件）</a>'+''.join(cards)
    (web/'index.html').write_text(page,encoding='utf-8')
    write_json(batch/'delivery.json',{'archive':str(archive),'sha256':sha256(archive),'files':files,
        'page':'/static/batches/'+batch.name+'/index.html','qwen_called':False})
    print('PACKAGED',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['candidates','extra','prepare','export']);p.add_argument('--batch',required=True,type=Path)
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--selections',type=Path);p.add_argument('--case');p.add_argument('--level-only',action='store_true');a=p.parse_args()
    batch=a.batch.resolve();batch.relative_to(Path.cwd()/'industrial_swap_v2/outputs')
    if a.action=='candidates':candidates(batch,a.gpu)
    elif a.action=='extra':extra_views(batch,a.case,a.level_only)
    elif a.action=='prepare':prepare(batch,read_json(a.selections))
    else:export(batch)
