"""Inspect saved M1 selections on registered photographs; no localization or IoU.

Camera search uses actual occlusion-aware Gaussian membership contributions.
Display overlays are explicitly contribution maps over original RGB, not masks
invented from text or new segmentation. Source Gaussian data remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4'
DATA = BASE / 'F08_localization_revised'
OUT = BASE / 'F08_candidate_selection'
os.environ.update(ROOT=str(ROOT), GSLE_OUTPUT=str(OUT/'display_adapter'),
                  GSLE_RECORDS=str(OUT/'display_records'), SWAP_DISABLE_QWEN='1')
sys.path.insert(0, str(ROOT/'gsle'))
import torch
import common as c


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''): h.update(chunk)
    return h.hexdigest()
def font(size): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def frame_id(name): return int(Path(name).stem.rsplit('_',1)[-1])
def bbox(mask):
    yy,xx=np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None


def load_data():
    scene_info=read(DATA/'sources.json')['scene']
    assert sha(scene_info['path'])==scene_info['sha256']
    scene=c.GaussianData.load(scene_info['path'])
    rows=read(DATA/'case_index.json')['cases']
    return scene,scene_info,rows,c.source_cameras(938)


def markup(rgb,contribution):
    # Density is actual full-scene Gaussian visibility, retaining occlusions.
    a=contribution[0].clip(0,1)[...,None]*.60
    image=np.asarray(rgb).astype(float)*(1-a)+np.array([255,30,25])*a
    result=Image.fromarray(np.round(image).clip(0,255).astype(np.uint8))
    from scipy.ndimage import binary_erosion
    pm=contribution[1]>.05
    edge=pm & ~binary_erosion(pm,iterations=2)
    arr=np.asarray(result).copy();arr[edge]=[0,220,235]
    result=Image.fromarray(arr)
    d=ImageDraw.Draw(result)
    for ch,label,color in [(0,'T: text-only',(255,50,40)),(1,'P: intended / photo-guided',(0,230,245))]:
        b=bbox(contribution[ch]>.05)
        if not b:continue
        x=max(0,min(b[0],rgb.width-255));y=max(0,b[1]-24)
        d.rectangle([x,y,x+255,y+23],fill=(20,20,20))
        d.text((x+3,y+2),label,font=font(16),fill=color)
    return result


@torch.no_grad()
def search():
    scene,scene_info,rows,cameras=load_data()
    xyz=scene.xyz.cpu().numpy()
    OUT.mkdir(parents=True,exist_ok=True)
    proposals=[]
    # These three have category-compatible original descriptions. M1-10 is excluded.
    for number in [3,6,1]:
        row=next(r for r in rows if r['catalog_number']==number)
        folder=DATA/'cases'/row['m1_case_id'];dest=OUT/'camera_search'/row['label']
        dest.mkdir(parents=True,exist_ok=True)
        ids=[np.load(folder/(method+'_indices.npy')) for method in ['text_only','photo_assisted']]
        points=[xyz[i[::max(1,len(i)//700)]] for i in ids]
        ranked=[]
        for cam in cameras:
            scores=[];fracs=[]
            for pts in points:
                uv,z=cam.project(pts)
                valid=(z>.01)&(uv[:,0]>12)&(uv[:,0]<cam.width-12)&(uv[:,1]>12)&(uv[:,1]<cam.height-12)
                fracs.append(float(valid.mean()))
                if valid.sum()<5:scores.append(0.);continue
                lo,hi=np.quantile(uv[valid],[.02,.98],axis=0)
                scores.append(float(np.prod(hi-lo)))
            ranked.append((min(scores)*min(fracs)**4,cam))
        ranked.sort(key=lambda q:q[0],reverse=True)
        shortlist=[]
        for score,cam in ranked:
            if score<=0:continue
            if all(abs(frame_id(cam.name)-frame_id(p.name))>=6 for p in shortlist):shortlist.append(cam)
            if len(shortlist)>=18:break
        # Include the actual input reference, historical inspection and native text view.
        required=[v['camera']['name'] for v in row['views']]+[row['text_native_camera']['name']]
        for name in required:
            if name not in {v.name for v in shortlist}:shortlist.append(next(v for v in cameras if v.name==name))
        flags=torch.zeros_like(scene.xyz)
        flags[torch.as_tensor(ids[0],device='cuda'),0]=1
        flags[torch.as_tensor(ids[1],device='cuda'),1]=1
        records=[]
        for cam in shortlist:
            contrib=c.raster(cam,scene,flags).cpu().numpy()
            masks=contrib[:2]>.05
            counts=[int(m.sum()) for m in masks]
            boxes=[bbox(m) for m in masks]
            # Small-object visibility is prioritized; no inferred correctness label.
            score=float(min(counts)**.5)
            record=dict(camera=cam.record(),visible_pixels=counts,bboxes=boxes,score=score,
                        original_photo=str(ROOT/'data/industrial/images'/cam.name))
            write(dest/(Path(cam.name).stem+'.json'),record)
            rgb=Image.open(record['original_photo']).convert('RGB').resize((cam.width,cam.height),Image.Resampling.LANCZOS)
            markup(rgb,contrib).save(dest/(Path(cam.name).stem+'.jpg'),quality=94)
            records.append(record)
        records.sort(key=lambda r:r['score'],reverse=True)
        sheet=Image.new('RGB',(1876,4*566),'white');d=ImageDraw.Draw(sheet)
        for k,rec in enumerate(records[:8]):
            x=(k%2)*938;y=(k//2)*566
            text=f"{row['label']} | {rec['camera']['name']} | visible T/P: {rec['visible_pixels']}"
            d.text((x+6,y+5),text,font=font(20),fill='black')
            sheet.paste(Image.open(dest/(Path(rec['camera']['name']).stem+'.jpg')),(x,y+34))
        sheet.save(dest/'camera_contact.jpg',quality=95)
        write(dest/'rankings.json',dict(case=row['label'],description=row['description'],
             m1_case_id=row['m1_case_id'],camera_count=len(cameras),screened_views=len(records),rankings=records))
        proposals.append(dict(case=row['label'],m1_case_id=row['m1_case_id'],top_views=records[:8]))
        print(row['label'],[(r['camera']['name'],r['visible_pixels']) for r in records[:8]],flush=True)
    write(OUT/'camera_search/manifest.json',dict(scene=scene_info,candidates=proposals,
         localization_rerun=False,iou_computed=False,selection_indices_unchanged=True,
         camera_search='All registered camera poses screened by saved selections; shortlist ranked with full-scene rendered membership contribution.'))


CHOICES=[
    dict(number=3,view='frame_000232.jpg',category='Portable industrial air cooler',
         title='Similar coolers: noticeboard side vs. central aisle',
         reason='Both actual devices are gray portable air coolers. The intended unit beside the green noticeboard and the text-selected unit in the aisle are complete and spatially distinct in one registered photograph.',
         caveat='The two devices have different image sizes; the shared crop and separate detail images expose this honestly.',
         reason_zh='两台均为灰色便携式工业冷风机；目标位于绿色公告板旁，Text-only 选中通道中央另一台。同一张注册照片中两台完整入镜，最直观地体现同类别实例歧义。',
         caveat_zh='目标和错误实例在画面中的大小不同，已同时保留完整照片与各自局部放大图。'),
    dict(number=6,view='frame_000183.jpg',category='Portable industrial air cooler',
         title='Similar coolers: rack side vs. central aisle',
         reason='The user-designated rack-side unit and the text-selected central-aisle unit are both portable air coolers, visible together. A third similar cooler is also present near the noticeboard.',
         caveat='Foreground assembly fixtures partly occlude the lower portions; the photo-guided Gaussian boundary contains minor surrounding spill. This candidate shares the text-only cooler with candidate 01.',
         reason_zh='目标为储物架旁的冷风机，Text-only 选中通道中央冷风机。同框还可见公告板旁的另一台冷风机，能展示同类多实例场景。',
         caveat_zh='前景工装对下部有一定遮挡，照片辅助选中高斯边缘略有溢出；与候选 01 共用同一台 Text-only 错误实例，不是独立场景。'),
    dict(number=1,view='frame_000064.jpg',category='Automotive assembly workstation',
         title='White assembly workstations in separate production rows',
         reason='The original broad category describes both the user-designated black assembly on a white workstation and the other white automotive assembly carrier selected by text. Both are visible in the same photograph.',
         caveat='The devices differ in fixtures and configuration, and the wrong instance is farther away. The intended base is partly occluded. This is weaker visual ambiguity than the two cooler cases.',
         reason_zh='原文为白色汽车装配工位，可合理覆盖用户指定的黑色组件/白色底座工位与 Text-only 选中的另一台白色装配承载工位。同一张照片可看见两者。',
         caveat_zh='两台工位的夹具和上部配置不同，错误实例较远、目标底座局部受挡；属于同类别工位歧义，外观近似程度弱于前两个冷风机案例。')]


def context_crop(box, width, height, pad=.22, square=True):
    x0,y0,x1,y1=box;bw=x1-x0;bh=y1-y0
    if square:bw=bh=max(bw,bh)*(1+2*pad)
    else:bw*=1+2*pad;bh*=1+2*pad
    bw=min(width,bw);bh=min(height,bh)
    x=max(0,min(width-bw,(x0+x1-bw)/2));y=max(0,min(height-bh,(y0+y1-bh)/2))
    return [int(round(x)),int(round(y)),int(round(x+bw)),int(round(y+bh))]


def overlay(rgb, contribution, cyan=False, red=True):
    arr=np.asarray(rgb).astype(float)
    if red:
        a=contribution.clip(0,1)[...,None]*.60
        arr=arr*(1-a)+np.array([255.,30.,25.])*a
    arr=np.round(arr).clip(0,255).astype(np.uint8)
    if cyan:
        from scipy.ndimage import binary_erosion
        region=contribution>.05
        arr[region & ~binary_erosion(region,iterations=2)]=[0,220,235]
    return Image.fromarray(arr)


def place(canvas,path,rect):
    from PIL import ImageOps
    x,y,w,h=rect
    im=ImageOps.contain(Image.open(path).convert('RGB'),(w,h),Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))


@torch.no_grad()
def package():
    scene,scene_info,rows,cameras=load_data()
    camera_by_name={v.name:v for v in cameras}
    snapshot={scene_info['path']:sha(scene_info['path'])}
    records=[]
    for rank,choice in enumerate(CHOICES,1):
        row=next(r for r in rows if r['catalog_number']==choice['number'])
        original=Path(row['source_case']);source=DATA/'cases'/row['m1_case_id']
        dest=OUT/f'candidate_{rank:02d}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['crops','original_reference','evidence']:(dest/name).mkdir(exist_ok=True)
        reference=original/'reference'
        files={'reference_photo':reference/'original_photo.jpg','reference_sam':reference/'reference_mask.png',
               'reference_selection':reference/'selection.json','m1_job':original/'job.json',
               'confirmed_case':original/'confirmed_case.json',
               'photo_assisted_indices':source/'photo_assisted_indices.npy',
               'text_only_indices':source/'text_only_indices.npy',
               'selected_photo':ROOT/'data/industrial/images'/choice['view']}
        for k,p in files.items():snapshot[str(p)]=sha(p)
        assert read(files['m1_job'])['text']==row['description']
        for k,name in [('reference_photo','reference_original.jpg'),('reference_sam','sam_mask.png'),
                       ('reference_selection','selection.json'),('m1_job','job.json'),('confirmed_case','confirmed_case.json')]:
            shutil.copy2(files[k],dest/'original_reference'/name)
        for method in ['text_only','photo_assisted']:
            shutil.copy2(files[method+'_indices'],dest/'evidence'/(method+'_indices.npy'))
        shutil.copy2(source/'reference_selection.jpg',dest/'original_reference/sam_overlay.jpg')
        shutil.copy2(files['selected_photo'],dest/'selected_camera_original.jpg')
        cam=camera_by_name[choice['view']]
        raw=Image.open(files['selected_photo']).convert('RGB')
        rgb=raw.resize((cam.width,cam.height),Image.Resampling.LANCZOS)
        rgb.save(dest/'selected_camera_rgb.png')
        ids=[np.load(files[method+'_indices']) for method in ['text_only','photo_assisted']]
        flags=torch.zeros_like(scene.xyz)
        for ch,indices in enumerate(ids):flags[torch.as_tensor(indices,device='cuda'),ch]=1
        contribution=c.raster(cam,scene,flags).cpu().numpy()
        np.savez_compressed(dest/'evidence/visible_gaussian_contributions.npz',text_only=contribution[0],photo_assisted=contribution[1])
        masks=contribution[:2]>.05
        boxes=[bbox(m) for m in masks];cropboxes=[context_crop(b,cam.width,cam.height) for b in boxes]
        # Original photograph with both visible actual selections for review only.
        markup(rgb,contribution).save(dest/'both_instances_check.jpg',quality=96)
        overlay(rgb,contribution[1],cyan=True,red=False).save(dest/'intended_target_check.png')
        for ch,method in enumerate(['text_only','photo_assisted']):
            tint=overlay(rgb,contribution[ch])
            tint.save(dest/(method+'_photo_projection.png'))
            tint.crop(cropboxes[ch]).save(dest/'crops'/(method+'_selected_projection.png'))
            rgb.crop(cropboxes[ch]).save(dest/'crops'/(method+'_original_rgb.png'))
            Image.fromarray(masks[ch].astype(np.uint8)*255).save(dest/'evidence'/(method+'_visible_mask.png'))
            colors=scene.colors(cam.center)
            ti=torch.as_tensor(ids[ch],device='cuda')
            colors[ti]=.4*colors[ti]+.6*colors.new_tensor([1.,0.,0.])
            c.save_image(dest/(method+'_gaussian_render.png'),c.raster(cam,scene,colors))
        colors=scene.colors(cam.center)
        c.save_image(dest/'scene_gaussian_render.png',c.raster(cam,scene,colors))
        # Keep the actual input photo + SAM separate from the later display camera.
        refim=Image.open(reference/'original_photo.jpg').convert('RGB')
        refmask=np.asarray(Image.open(reference/'reference_mask.png'))>127
        refbox=context_crop(bbox(refmask),refim.width,refim.height)
        refim.crop(refbox).save(dest/'crops/reference_target_original.png')
        Image.open(source/'reference_selection.jpg').crop(refbox).save(dest/'crops/reference_sam_target.png')
        record=dict(rank=rank,directory=str(dest),m1_label=row['label'],m1_case_id=row['m1_case_id'],
            original_text=row['description'],category=choice['category'],reason=choice['reason'],caveat=choice['caveat'],
            reason_zh=choice['reason_zh'],caveat_zh=choice['caveat_zh'],
            input_reference_camera=row['reference_camera'],selected_display_camera=cam.record(),
            selected_photo_native_size=list(raw.size),reference_camera_changed_for_localization=False,
            display_camera_only=True,same_photograph_shows_both_instances=True,
            counts={'text_only_gaussians':len(ids[0]),'photo_assisted_gaussians':len(ids[1])},
            visible_selection_bounds={'text_only':boxes[0],'photo_assisted':boxes[1]},
            crop_boxes={'text_only':cropboxes[0],'photo_assisted':cropboxes[1],'reference_target':refbox},
            files={k:dict(path=str(p),sha256=snapshot[str(p)]) for k,p in files.items()},
            evaluation_source=str(source/'evaluation.json'),human_approval_pending=True,
            labels='T: text-only selected wrong instance; P: intended instance identified from original SAM and saved photo-assisted selection.',
            rendering='Actual full-scene Gaussian membership contribution on registered RGB; separate full-scene Gaussian RGB highlights are also included. Cyan display contour is not independent GT.',
            no_new_localization=True,no_iou_computation=True)
        write(dest/'camera.json',cam.record());write(dest/'manifest.json',record)
        # Inspection sheet, not a manuscript F08 layout.
        board=Image.new('RGB',(2100,1300),'white');d=ImageDraw.Draw(board)
        d.text((16,8),f"Candidate {rank:02d} | {row['label']} | {choice['title']}",font=font(26),fill='black')
        d.text((16,43),'Original Text A: "'+row['description']+'"',font=font(23),fill='#333333')
        labels=['Original reference + user SAM','Chosen registered photo: both instances','Text-only Gaussian projection','Photo-guided Gaussian projection']
        paths=[dest/'original_reference/sam_overlay.jpg',dest/'both_instances_check.jpg',dest/'text_only_photo_projection.png',dest/'photo_assisted_photo_projection.png']
        for k,(label,path) in enumerate(zip(labels,paths)):
            x=(k%2)*1050;y=85+(k//2)*335
            d.text((x+10,y),label,font=font(23),fill='black');place(board,path,(x+6,y+32,1038,292))
        labels=['Intended target: original RGB','Wrong instance: original RGB','Text-only selected Gaussians','Photo-guided selected Gaussians']
        paths=[dest/'crops/photo_assisted_original_rgb.png',dest/'crops/text_only_original_rgb.png',dest/'crops/text_only_selected_projection.png',dest/'crops/photo_assisted_selected_projection.png']
        for k,(label,path) in enumerate(zip(labels,paths)):
            d.text((k*525+10,770),label,font=font(21),fill='black');place(board,path,(k*525+10,807,505,440))
        d.text((14,1255),f"Reference: {row['reference_camera']['name']} | Display: {cam.name} | Existing M1 selections; no IoU recomputation",font=font(19),fill='#444444')
        board.save(dest/'inspection_sheet.jpg',quality=96)
        records.append(record)
        print('Packaged',row['label'],cam.name,raw.size,flush=True)

    assert all(sha(p)==digest for p,digest in snapshot.items())
    # Archive audit conclusions for all ten; never silently count rejected cases.
    decisions={1:'Candidate 03: compatible automotive-workstation category; weaker visual similarity and greater separation.',
        2:'Excluded: both methods select the intended instance; no wrong-vs-correct example.',
        3:'Candidate 01: strongest same-category cooler ambiguity with both devices completely in view.',
        4:'Excluded: both methods select the intended cooler. Same physical cooler as M1-09.',
        5:'Excluded from final three: intended large wheeled frame and text-selected assembly carrier have different structures; cooler cases and M1-01 provide clearer category agreement.',
        6:'Candidate 02: rack-side vs. aisle cooler; foreground occlusion and boundary spill are disclosed.',
        7:'Excluded: stored crate description imperfectly matches metal-rack assembly; both select same target.',
        8:'Excluded: stored belt-conveyor description conflicts with selected assembly workstation.',
        9:'Excluded: both methods select the intended cooler; duplicate physical target of M1-04.',
        10:'Excluded: stored air-cooler text conflicts with the visible narrow wheeled slatted frame.'}
    audit=[dict(m1_label=r['label'],m1_case_id=r['m1_case_id'],text=r['description'],
                reference=r['reference_camera']['name'],decision=decisions[r['catalog_number']]) for r in rows]
    manifest=dict(task='F08 candidate inspection only; no manuscript figure updated',
        candidates=records,recommended_candidate='candidate_01',scene=scene_info,all_ten_case_audit=audit,
        camera_search_manifest=read(OUT/'camera_search/manifest.json'),
        integrity=dict(original_files_unchanged=True,source_hashes=snapshot),
        original_m1_and_text_only_indices_reused=True,no_iou_recomputed=True,no_m2_m3_m4=True,
        manual_confirmation_required_for_final_figure=True,
        note='These are three selected examples from one scene, not three independent scenes; candidates 01 and 02 share the same text-selected cooler. No aggregate statistics are changed.')
    write(OUT/'selection_manifest.json',manifest)
    overview=Image.new('RGB',(2400,3*455+75),'white');d=ImageDraw.Draw(overview)
    heads=['Original reference + SAM','Both instances in one registered photo','Text-only: wrong instance','Photo-guided: intended instance']
    for i,s in enumerate(heads):d.text((i*600+8,12),s,font=font(22),fill='black')
    for i,r in enumerate(records):
        dest=Path(r['directory']);y=55+i*455
        d.text((10,y),f"Candidate {i+1:02d} | {r['m1_label']} | {r['original_text']} | {r['selected_display_camera']['name']}",font=font(22),fill='black')
        paths=[dest/'original_reference/sam_overlay.jpg',dest/'both_instances_check.jpg',dest/'crops/text_only_selected_projection.png',dest/'crops/photo_assisted_selected_projection.png']
        for j,p in enumerate(paths):place(overview,p,(j*600+5,y+40,590,395))
    overview.save(OUT/'candidate_overview.jpg',quality=96)
    summary=['# F08 实例歧义候选筛选','',
        '本次仅筛选照片与整理已有 M1 定位检查材料。没有修改 F08 论文图，没有重新定位、计算 IoU 或运行 M2–M4。',
        '完整核对了 10 个正式 M1 案例。推荐 **candidate_01 / M1-03**。三个候选均可在选出的同一张原始注册照片中看到目标与 Text-only 实际选中的另一实例。','',
        '## 显示与证据约定','',
        '- 原始输入照片、用户 SAM 与三维选中索引均保留；另选的相机只用于检查，没有替换定位输入。',
        '- 红色来自实际选中高斯在完整场景中的可见贡献，保留遮挡与透明度；检查图叠加到原始照片上。另保存了直接高斯场景渲染图。',
        '- 同框检查图中的青色为已保存照片辅助高斯集合的可见轮廓，用于指出目标，不是独立 GT 标注。',
        '- 原始照片不作修补。局部放大图只做裁剪和等比例缩放。没有计算或更新任何 IoU。','']
    for r in records:
        summary += [f"## candidate_{r['rank']:02d} · {r['m1_label']}",'',
            f"- M1 Case ID：`{r['m1_case_id']}`",f"- 原始 Text A：**{r['original_text']}**",f"- 类别：{r['category']}",
            f"- 原始参考照片：`{r['input_reference_camera']['name']}`",f"- 推荐检查照片：`{r['selected_display_camera']['name']}`",f"- 推荐理由：{r['reason_zh']}",f"- 局限：{r['caveat_zh']}",
            f"- 检查图：`candidate_{r['rank']:02d}/inspection_sheet.jpg`；完整原图：`selected_camera_original.jpg`；局部图：`crops/`。",'']
    summary += ['## 未采用的案例','',
        '- M1-02、M1-04、M1-09：两种方法均选对，不能作为“纯文本选错、照片引导选对”的例子；M1-04/09 还是同一物理设备。',
        '- M1-07、M1-08、M1-10：文本类别与实际选中结构存在不符，排除。',
        '- M1-05：同为有轮工业承载结构，但具体设备结构差异较大，优先级低于所选三个案例。','',
        '## 使用边界','',
        '以上为基于真实照片、用户确认的 SAM 与已保存高斯结果的筛选意见，等待用户确认。候选 01/02 共享同一台 Text-only 错误冷风机，三例全部来自同一个工业场景，不应当作独立场景样本。',
        '本次没有改变此前实例选择统计，也未把新的相机搜索作为新的定位结果。','']
    (OUT/'candidate_summary.md').write_text('\n'.join(summary),encoding='utf-8')
    import html
    page=['<!doctype html><meta charset="utf-8"><title>F08 candidate selection</title><style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:20px}img{max-width:100%}a{color:#165cb2}h2{margin-top:40px}</style>',
        '<h1>F08 候选检查：等待人工确认</h1><p>推荐候选 01 / M1-03。以下仅为照片与定位检查，不是最终论文图。</p>',
        '<p><a href="candidate_summary.md">候选说明</a> · <a href="selection_manifest.json">完整来源记录</a> · <a href="candidate_overview.jpg">总览高清图</a></p><img src="candidate_overview.jpg">']
    for r in records:
        folder=f"candidate_{r['rank']:02d}"
        page += [f"<h2>{folder} · {r['m1_label']}</h2><p><i>{html.escape(r['original_text'])}</i></p><p>{r['reason_zh']}</p><p>局限：{r['caveat_zh']}</p>",
            f'<p><a href="{folder}/selected_camera_original.jpg">完整注册照片</a> · <a href="{folder}/original_reference/reference_original.jpg">实际 M1 输入照片</a> · <a href="{folder}/inspection_sheet.jpg">高清检查图</a> · <a href="{folder}/crops/">局部放大图</a></p><img src="{folder}/inspection_sheet.jpg">']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    (OUT/'code').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    print('Completed three candidate packages; existing experiment files unchanged.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--search',action='store_true')
    parser.add_argument('--package',action='store_true')
    args=parser.parse_args()
    if args.search:search()
    if args.package:package()
