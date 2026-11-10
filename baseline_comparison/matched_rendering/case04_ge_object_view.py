"""Case04 GE correction: center the camera on its actual generated Gaussians."""
import math
import shutil
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import render as r
import localization_red as lr

BASE=r.BASE/'target_centered_views'
PREV=BASE/'case04_view_adjust_v2'
OUT=BASE/'case04_ge_object_view_v3'
CAPTION='GE camera targets its generated object | GSLE = Ours view | Red annotation opacity: 60%'


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
    previous_center=np.asarray(report['gaussianeditor_search']['selected']['center'])
    # Keep the previous near-Ours direction initially, but move the look-at point
    # to the actual generated object. Old localization does not enter the score.
    bounds=r.corners(info['geometry']);lo,hi=.05,30.
    for _ in range(25):
        distance=(lo+hi)/2
        cam=r.camera_at(ref,previous_center,center,distance,up)
        xy,z=cam.project(bounds)
        size=np.ptp(xy,axis=0).max()/1024 if (z>0).all() else 1e6
        if size>.55:lo=distance
        else:hi=distance
    initial=(lo+hi)/2;trials=[]
    def evaluate(distance,yaw=0,pitch=0):
        cam=r.camera_at(ref,previous_center,center,distance,up,yaw,pitch,'case04_ge_generated_object')
        m=r.candidate_metrics(cam,obj,final,flags,res=224)
        score=40*max(0,.985-m['capture'])+10*max(0,.85-m['visibility'])
        score+=1.7*abs(math.log(max(m['size'],.001)/.55))+2*max(0,.65-m['scene_coverage'])
        score+=.012*abs(yaw)+.02*abs(pitch)
        if m['bbox']:
            x0,y0,x1,y1=m['bbox'];margin=min(x0,y0,224-x1,224-y1)
            score+=2*max(0,1-margin/8)
        item=dict(camera=cam.record(),score=score,yaw_offset=yaw,pitch_offset=pitch,**m)
        trials.append(item);return item
    for factor in [.8,1.,1.25,1.6]:evaluate(initial*factor)
    best=min(trials,key=lambda x:x['score'])
    if best['visibility']<.8:
        for yaw,pitch in [(-12,0),(12,0),(-24,0),(24,0),(0,8),(0,-8)]:evaluate(initial,yaw,pitch)
        best=min(trials,key=lambda x:x['score'])
    cam=r.c.Camera.from_record(best['camera'])
    views={'main':cam}
    for n,yaw in [(1,-20),(2,20)]:views[f'aux{n}']=r.camera_at(cam,center,center,np.linalg.norm(cam.center-center),up,yaw,0,f'ge_object_aux{n}')
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
            method['shared_crop']=r.square_crop(stats['bbox'],1024)
        cameras['methods']['gaussianeditor'][name]=dict(localization_camera=view.record(),final_camera=view.record())
        method['views'][name]=dict(localization_final_camera_identical=True,generated_object=stats,
            visible_localization=r.alpha_stats(contribution.cpu().numpy()))
        print('GE',name,'complete',flush=True)
    selection=dict(target_center=center.tolist(),center_source=info['object_path'],selected=best,candidates=trials,
        policy='Look at actual generated-object center; old localization is not a framing target. Preserve prior near-Ours direction unless visibility requires a small change.')
    report.update(revision=OUT.name,previous_revision=str(PREV),gaussianeditor_search=selection,
        input_sha256=hashes,other_method_image_sha256=copied)
    report['methods']['gaussianeditor']=method
    cameras['policy']='GE targets its actual generated object; GSLE/DGE/Ours retain the accepted v2 cameras; each localization/final pair shares one camera.'
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
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>case04：GE 面向生成物体</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:26px 0}</style><h1>case04：GaussianEditor 面向其实际生成物体</h1><p>GaussianEditor 以已有 Add object_world.ply 中的新设备位置为取景中心，优先保证生成物体的展示。红色定位仍显示全部原始定位索引，并与最终结果使用同一相机。定位位置与生成位置的偏差如实保留。</p><p>GSLE 继续共用 Ours 相机；DGE、GSLE、Ours 的相机和图片逐文件保持上一版不变。红色标注为 60% 红色 + 40% 原色。没有训练或修改任何三维模型。</p><p><a href="../index.html">全部案例</a> | <a href="camera_params.json">相机参数</a> | <a href="report.json">检查记录</a></p>'''
    for name,label in [('comparison_paired','定位 / 最终结果'),('comparison_final','最终场景'),('comparison_localization','半透明定位'),('comparison_crops','局部放大'),('comparison_aux1','辅助视角 1'),('comparison_aux2','辅助视角 2')]:
        page+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (OUT/'index.html').write_text(page+'</html>',encoding='utf-8')
    gallery=(BASE/'index.html').read_text(encoding='utf-8').replace('case04_view_adjust_v2/',OUT.name+'/')
    gallery=gallery.replace('GaussianEditor 接近 Ours 方向并显示自身定位。','GaussianEditor 朝向其实际生成物体。')
    for stage in ['localization','final']:
        old_name=f'overview_{stage}_case00_case02_case04_v2.png';new_name=f'overview_{stage}_case00_case02_case04_v3.png'
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
    print('COMPLETE case04 GE object-centered',best,flush=True)


if __name__=='__main__':main()
