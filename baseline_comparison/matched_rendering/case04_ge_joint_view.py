"""Case04 GE camera revision: jointly frame localization and generated object."""
import math
import shutil
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import render as r
import localization_red as lr
import case04_joint_framing as joint

BASE=r.BASE/'target_centered_views'
PREV=BASE/'case04_ge_object_view_v3'
OUT=BASE/'case04_ge_joint_view_v4'
CAPTION='GE: localization AND generated object inside the same view | Other methods unchanged'


@torch.no_grad()
def main():
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    assert not (OUT/'report.json').exists(),'Preserve completed revision'
    OUT.mkdir(parents=True,exist_ok=True)
    report=r.read(PREV/'report.json');cameras=r.read(PREV/'camera_params.json')
    spec=r.read(r.BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']=='case04')
    info=r.read(r.BASE/'matched_views/case04/camera_params.json')['methods']['gaussianeditor']['target']
    paths=[Path(spec['source_scene']),Path(info['scene_path']),Path(info['object_path']),lr.index_path(row,'gaussianeditor')]
    hashes={str(p):r.sha(p) for p in paths}
    original=r.c.GaussianData.load(spec['source_scene']);final=r.c.GaussianData.load(info['scene_path'])
    obj,vis_scene,flags=r.flag_data(dict(info),final,original)
    center=np.asarray(info['center']);up=r.c.normalized(np.asarray(r.read(row['target_record'])['up']))
    ref=r.c.Camera.from_record(cameras['methods']['gaussianeditor']['main']['final_camera'])
    cam,joint_report=joint.choose(original,final,obj,flags,ref,center,up,OUT)
    best=joint_report['selected'];trials=joint_report['candidates']
    views={'main':cam}
    for name,record in joint_report['auxiliary_views'].items():views[name]=r.c.Camera.from_record(record['camera'])
    # Preserve every image and camera belonging to the three unchanged methods.
    copied={}
    for p in PREV.rglob('*.png'):
        if p.name.startswith(('dge_','gsle_','ours_')) or p.relative_to(PREV)==Path('original.png'):
            dest=OUT/p.relative_to(PREV);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            copied[str(p.relative_to(PREV))]=r.sha(p)
    ids=np.load(lr.index_path(row,'gaussianeditor'));idx=torch.as_tensor(ids,device='cuda')
    loc_flags=torch.zeros_like(original.xyz);loc_flags[idx]=1
    method=dict(localization_indices=str(lr.index_path(row,'gaussianeditor')),selected_gaussians=len(ids),
        final_scene=info['scene_path'],views={})
    for name,view in views.items():
        folder=OUT/'views'/name;folder.mkdir(parents=True,exist_ok=True)
        colors=original.colors(view.center);source=r.c.raster(view,original,colors)
        colors[idx]=.4*colors[idx]+.6*colors.new_tensor([1.,0.,0.])
        red=r.c.raster(view,original,colors);rgb=r.c.render(view,final,aux=False)['rgb']
        contribution=r.c.raster(view,original,loc_flags)[0]
        a=r.object_alpha(view,obj);stats=r.alpha_stats(a)
        for suffix,image in [('original',source),('localization',red),('final',rgb),('localization_mask',contribution[None]),('object_alpha',torch.from_numpy(a)[None])]:
            r.c.save_image(folder/('gaussianeditor_'+suffix+'.png'),image)
        if name=='main':
            for suffix in ['localization','final']:shutil.copy2(folder/('gaussianeditor_'+suffix+'.png'),OUT/('gaussianeditor_'+suffix+'.png'))
            local_box=r.alpha_stats(contribution.cpu().numpy())['bbox']
            box=stats['bbox']
            if local_box and box:box=[min(box[0],local_box[0]),min(box[1],local_box[1]),max(box[2],local_box[2]),max(box[3],local_box[3])]
            method['shared_crop']=r.square_crop(box or local_box,1024)
        cameras['methods']['gaussianeditor'][name]=dict(localization_camera=view.record(),final_camera=view.record())
        method['views'][name]=dict(localization_final_camera_identical=True,generated_object=stats,
            visible_localization=r.alpha_stats(contribution.cpu().numpy()))
        print('GE',name,'complete',flush=True)
    selection=joint_report
    report.update(revision=OUT.name,previous_revision=str(PREV),gaussianeditor_search=selection,
        input_sha256=hashes,other_method_image_sha256=copied)
    report['methods']['gaussianeditor']=method
    cameras['policy']='GE simultaneously frames its original localized target and actual generated object. Other methods remain unchanged. Each localization/final pair uses identical cameras.'
    for p,h in hashes.items():assert r.sha(p)==h
    for p,h in copied.items():assert r.sha(OUT/p)==h
    report['all_input_files_unchanged']=True;report['dge_gsle_ours_images_unchanged']=True
    r.save(OUT/'report.json',report);r.save(OUT/'camera_params.json',cameras);r.save(OUT/'ge_camera_search.json',selection)
    for stage in ['localization','final']:
        lr.board(OUT/('comparison_'+stage+'.png'),[[('Original',OUT/'original.png',None)]+
            [(label,OUT/(m+'_'+stage+'.png'),None) for m,label in zip(lr.METHODS,lr.LABELS)]],
            'case04 | '+stage,CAPTION)
    for name,cropped in [('comparison_paired',False),('comparison_crops',True)]:
        lr.board(OUT/(name+'.png'),[[(label+' / '+stage,OUT/(m+'_'+stage+'.png'),report['methods'][m]['shared_crop'] if cropped else None)
            for m,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],
            'case04 | Same-camera localization / final',CAPTION)
    for n in (1,2):
        lr.board(OUT/f'comparison_aux{n}.png',[[(label+' / '+stage,OUT/'views'/f'aux{n}'/(m+'_'+stage+'.png'),None)
            for m,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],f'case04 | Auxiliary {n}',CAPTION)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>case04：GE 分割与生成同时入镜</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:26px 0}</style><h1>case04：GaussianEditor 分割区域与生成物体同时入镜</h1><p>相机同时覆盖其实际定位区域与实际生成物体，并为两处保留画面边距。上排显示原场景中的红色定位，下排显示最终场景，二者相机完全一致。两处真实空间位置没有移动或对齐修改。</p><p>GSLE 继续共用 Ours 相机；DGE、GSLE、Ours 的图片和相机保持上一版不变。红色标注为 60% 红色 + 40% 原色，没有训练或修改任何三维模型。</p><p><a href="../index.html">全部案例</a> | <a href="camera_params.json">相机参数</a> | <a href="joint_framing_search.json">双区域选角记录</a> | <a href="report.json">检查记录</a></p>'''
    for name,label in [('comparison_paired','定位 / 最终结果'),('comparison_final','最终场景'),('comparison_localization','半透明定位'),('comparison_crops','局部放大'),('comparison_aux1','辅助视角 1'),('comparison_aux2','辅助视角 2')]:
        page+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (OUT/'index.html').write_text(page+'</html>',encoding='utf-8')
    gallery=(BASE/'index.html').read_text(encoding='utf-8').replace('case04_ge_object_view_v3/',OUT.name+'/')
    gallery=gallery.replace('GaussianEditor 朝向其实际生成物体。','GaussianEditor 的分割区域与生成物体同时入镜。')
    for stage in ['localization','final']:
        old_name=f'overview_{stage}_case00_case02_case04_v3.png';new_name=f'overview_{stage}_case00_case02_case04_v4.png'
        im=Image.open(BASE/old_name).convert('RGB');part=Image.open(OUT/('comparison_'+stage+'.png'))
        im.paste(part.crop((0,80,2500,620)),(0,1170))
        ImageDraw.Draw(im).text((12,1208),'case04',font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
        im.save(BASE/new_name,dpi=(300,300));gallery=gallery.replace(old_name,new_name)
    (BASE/'index.html').write_text(gallery,encoding='utf-8')
    with zipfile.ZipFile(OUT/(OUT.name+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.json','.html'):
                if p.name=='index.html':z.writestr('index.html',p.read_text(encoding='utf-8').replace('<a href="../index.html">全部案例</a> | ',''))
                else:z.write(p,p.relative_to(OUT))
    print('COMPLETE case04 GE joint framing',best,flush=True)


if __name__=='__main__':main()
