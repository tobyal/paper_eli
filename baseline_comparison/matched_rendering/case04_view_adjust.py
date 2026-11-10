"""Case04 selective display revision: exact Ours cameras and 60% red annotation.

Annotation transparency is a color blend during rasterization. Physical Gaussian
opacity, geometry, localization indices and final scene appearance are unchanged.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import render as r
import localization_red as lr
import case04_camera_search as search

BASE=r.BASE/'target_centered_views'
OUT=BASE/'case04_view_adjust_v2'
CAPTION='GSLE = Ours view | GE: visible localization, near-Ours direction | Red annotation opacity: 60%'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-camera',action='store_true');args=parser.parse_args()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    assert not (OUT/'report.json').exists() or args.refresh_camera,'Preserve completed display revision'
    OUT.mkdir(parents=True,exist_ok=True)
    previous=r.read(BASE/'case04/report.json');params=r.read(BASE/'case04/camera_params.json')
    spec=r.read(r.BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']=='case04')
    views={name:r.c.Camera.from_record(v['final_camera']) for name,v in params['methods']['ours']['views'].items()}
    paths={Path(spec['source_scene']),BASE/'case04/report.json',BASE/'case04/camera_params.json'}
    for method in lr.METHODS:
        paths.update([Path(previous['methods'][method]['final_scene']),lr.index_path(row,method)])
    paths.update([r.BASE/'gaussianeditor_native/case04/add/object_world.ply',r.BASE/'gaussianeditor_native/case04/anchor_camera.json'])
    before={str(p):r.sha(p) for p in paths}
    original=r.c.GaussianData.load(spec['source_scene'])
    ge_scene=r.c.GaussianData.load(previous['methods']['gaussianeditor']['final_scene'])
    center=np.asarray(r.read(row['target_record'])['center'])
    up=r.c.normalized(np.asarray(r.read(row['target_record'])['up']))
    if args.refresh_camera:
        ge_search=r.read(OUT/'ge_camera_search.json')
        for candidate in ge_search['candidates']:candidate['score']=search.score_record(candidate)
        selected=min(ge_search['candidates'],key=lambda x:x['score'])
        ge_search['selected']=selected;ge_camera=r.c.Camera.from_record(selected['camera'])
        own_angle=r.orientation(-ge_camera.R[2],up);ours_angle=r.orientation(-views['main'].R[2],up)
        ge_search['camera_optical_yaw_difference']=r.delta_angle(own_angle['yaw'],ours_angle['yaw'])
        ge_search['camera_optical_pitch_difference']=own_angle['pitch']-ours_angle['pitch']
        ge_search['framing_revision']='Require image-edge margin for both localization and generated object; reuse cached candidates.'
        r.save(OUT/'ge_camera_search.json',ge_search)
    else:
        ge_camera,ge_search=search.choose(original,ge_scene,views['main'],center,up,OUT)
    ge_center=np.asarray(ge_search['selected']['center'])
    ge_views={'main':ge_camera}
    for n,yaw in [(1,-20),(2,20)]:
        ge_views[f'aux{n}']=r.camera_at(ge_camera,ge_center,ge_center,np.linalg.norm(ge_camera.center-ge_center),up,yaw,0,f'case04_ge_aux{n}')
    del ge_scene;torch.cuda.empty_cache()
    report=dict(case='case04',revision='case04_view_adjust_v2',annotation_red_opacity=.6,
        annotation_original_color_weight=.4,physical_gaussian_opacity_unchanged=True,
        full_native_localization_indices=True,gsle_shares_ours_cameras=True,
        gaussianeditor_search=ge_search,dge_ours_camera_unchanged=True,
        training_steps=0,ply_writes=0,methods={},input_sha256=before,
        common_crop=previous['methods']['ours']['shared_crop'])
    cameras=dict(case='case04',reference=str(BASE/'case04/camera_params.json'),
        policy='GSLE uses exact Ours cameras. GE chooses a visible own-localization view near Ours direction. DGE and Ours cameras unchanged. Each localization/final pair shares one camera.',methods={})
    for method in lr.METHODS:
        if method=='gaussianeditor':method_views=ge_views
        elif method=='dge':method_views={name:r.c.Camera.from_record(v['final_camera']) for name,v in params['methods']['dge']['views'].items()}
        else:method_views=views
        ids=np.load(lr.index_path(row,method));idx=torch.as_tensor(ids,device='cuda')
        assert len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(original.xyz)
        scene=r.c.GaussianData.load(previous['methods'][method]['final_scene'])
        flags=torch.zeros_like(original.xyz);flags[idx]=1
        method_report=dict(localization_indices=str(lr.index_path(row,method)),
            selected_gaussians=len(ids),final_scene=previous['methods'][method]['final_scene'],views={},
            shared_crop=previous['methods']['ours' if method in ['ours','gsle'] else method]['shared_crop'])
        cameras['methods'][method]={}
        for name,cam in method_views.items():
            folder=OUT/'views'/name;folder.mkdir(parents=True,exist_ok=True)
            color=original.colors(cam.center)
            source=r.c.raster(cam,original,color)
            color[idx]=.4*color[idx]+.6*color.new_tensor([1.,0.,0.])
            localization=r.c.raster(cam,original,color)
            final=r.c.render(cam,scene,aux=False)['rgb']
            contribution=r.c.raster(cam,original,flags)[0]
            outside=contribution==0
            error=float((localization-source).abs()[:,outside].max()) if outside.any() else 0.
            assert error<2e-6,error
            r.c.save_image(folder/(method+'_original.png'),source)
            for suffix,image in [('localization',localization),('final',final),('localization_mask',contribution[None])]:
                r.c.save_image(folder/(method+'_'+suffix+'.png'),image)
            if name=='main':
                for suffix in ['localization','final']:
                    shutil.copy2(folder/(method+'_'+suffix+'.png'),OUT/(method+'_'+suffix+'.png'))
                if method=='ours':shutil.copy2(folder/(method+'_original.png'),OUT/'original.png')
                if method=='gaussianeditor':
                    box=r.alpha_stats(contribution.cpu().numpy())['bbox']
                    generated=ge_search['selected']['generated_object']['bbox']
                    if generated:
                        generated=[int(x*1024/224) for x in generated]
                        box=generated if box is None else [min(box[0],generated[0]),min(box[1],generated[1]),max(box[2],generated[2]),max(box[3],generated[3])]
                    method_report['shared_crop']=r.square_crop(box,1024)
            method_report['views'][name]=dict(nonselected_max_rgb_difference=error,
                visible_localization=r.alpha_stats(contribution.cpu().numpy()),
                exact_ours_camera=method in ['gsle','ours'],localization_final_camera_identical=True)
            cameras['methods'][method][name]=dict(localization_camera=cam.record(),final_camera=cam.record())
            print(method,name,'complete',flush=True)
        report['methods'][method]=method_report
        del scene,flags,color,source,localization,final,contribution
        torch.cuda.empty_cache()
    for p,h in before.items():assert r.sha(p)==h
    report['all_input_files_unchanged']=True
    # Exact camera reuse must also preserve DGE/Ours final rendered pixels.
    for method in ['dge','ours']:
        for name in views:
            old=BASE/'case04/views'/name/(method+'_final.png')
            new=OUT/'views'/name/(method+'_final.png')
            assert np.array_equal(np.asarray(Image.open(old)),np.asarray(Image.open(new)))
    report['dge_ours_final_pixels_unchanged']=True
    r.save(OUT/'report.json',report);r.save(OUT/'camera_params.json',cameras)
    for stage in ['localization','final']:
        lr.board(OUT/('comparison_'+stage+'.png'),[[('Original',OUT/'original.png',None)]+
            [(label,OUT/(method+'_'+stage+'.png'),None) for method,label in zip(lr.METHODS,lr.LABELS)]],
            'case04 | '+stage+' | Requested view adjustment',CAPTION)
    for name,crop in [('comparison_paired',None),('comparison_crops',report['common_crop'])]:
        lr.board(OUT/(name+'.png'),[[(label+' / '+stage,OUT/(method+'_'+stage+'.png'),
            (report['methods'][method].get('shared_crop',previous['methods'][method]['shared_crop']) if crop is not None else None))
            for method,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],
            'case04 | Same-camera localization / final',CAPTION)
    for n in (1,2):
        lr.board(OUT/f'comparison_aux{n}.png',[[(label+' / '+stage,OUT/'views'/f'aux{n}'/(method+'_'+stage+'.png'),None)
            for method,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],
            f'case04 | Paired auxiliary view {n}',CAPTION)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>case04 视角微调</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:26px 0}</style><h1>case04：GSLE 统一 Ours，GaussianEditor 接近 Ours 方向并显示自身定位</h1><p>GSLE 的主视角与辅助视角完整复用 Ours 相机。GaussianEditor 以自身原生参考视角中可见贡献最大的定位连通区域为取景中心，在接近 Ours 方向的候选中检查实际定位可见性及生成物体可见性。所有定位索引仍保留在红色显示中，没有重新分割或移动物体。</p><p>DGE 和 Ours 相机及最终图像保持原样。每个方法的上下两排使用相同相机；不同方法之间相机可能不同。红色定位沿用 60% 红色 + 40% 原始颜色，仅改变标注显示，不改变高斯物理透明度。</p><p>只更新 case04 的展示。其他案例与所有最终 PLY 保持不变。</p><p><a href="../index.html">全部案例</a> | <a href="../case04/index.html">case04 上一版</a> | <a href="camera_params.json">相机参数</a> | <a href="ge_camera_search.json">GE 选角记录</a> | <a href="report.json">检查记录</a></p>'''
    for name,label in [('comparison_paired','定位 / 最终结果'),('comparison_localization','半透明红色定位'),
                       ('comparison_final','最终结果'),('comparison_crops','统一区域局部放大'),
                       ('comparison_aux1','共用辅助视角 1'),('comparison_aux2','共用辅助视角 2')]:
        page+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (OUT/'index.html').write_text(page+'</html>',encoding='utf-8')
    # Only the case04 section and case04 overview row change in the active gallery.
    # Old case04 images and old overviews remain available unchanged.
    page=(BASE/'index.html').read_text(encoding='utf-8')
    page=page.replace('href="case04/index.html"','href="case04_view_adjust_v2/index.html"')
    page=page.replace('href="case04/comparison_paired.png"','href="case04_view_adjust_v2/comparison_paired.png"')
    page=page.replace('src="case04/comparison_paired.png"','src="case04_view_adjust_v2/comparison_paired.png"')
    if 'case04 已更新：' not in page:
        page=page.replace('<h1>','<p><strong>case04 已更新：GSLE 共用 Ours 相机；GaussianEditor 接近 Ours 方向并显示自身定位。DGE、Ours 视角保持原样。</strong></p><h1>',1)
    for stage in ['localization','final']:
        # Start from the existing case00 revision, preserving that accepted row.
        im=Image.open(BASE/('overview_'+stage+'_case00_case02_v2.png')).convert('RGB')
        case=Image.open(OUT/('comparison_'+stage+'.png'))
        im.paste(case.crop((0,80,2500,620)),(0,1170))
        from PIL import ImageDraw,ImageFont
        ImageDraw.Draw(im).text((12,1208),'case04',font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
        name='overview_'+stage+'_case00_case02_case04_v2.png';im.save(BASE/name,dpi=(300,300))
        page=page.replace('href="overview_'+stage+'_case00_case02_v2.png"','href="'+name+'"')
    (BASE/'index.html').write_text(page,encoding='utf-8')
    # Standalone case04 archive has no external HTML dependencies.
    with zipfile.ZipFile(OUT/'case04_view_adjust_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.json','.html'):
                if p.name=='index.html':
                    text=p.read_text(encoding='utf-8')
                    text=text.replace('<a href="../index.html">全部案例</a> | ','').replace('<a href="../case04/index.html">case04 上一版</a> | ','')
                    z.writestr('index.html',text)
                else:z.write(p,p.relative_to(OUT))
    print('COMPLETE case04 revision',flush=True)


if __name__=='__main__':main()
