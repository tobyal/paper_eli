"""Case00-only display revision: exact Ours cameras and 60% red annotation.

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

BASE=r.BASE/'target_centered_views'
OUT=BASE/'case00_ours_view_v2'
CAPTION='Identical Ours camera for every method | Red annotation: 60% red + 40% original color'


@torch.no_grad()
def main():
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    assert not (OUT/'report.json').exists(),'Preserve completed display revision'
    OUT.mkdir(parents=True,exist_ok=True)
    previous=r.read(BASE/'case00/report.json');params=r.read(BASE/'case00/camera_params.json')
    spec=r.read(r.BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']=='case00')
    views={name:r.c.Camera.from_record(v['final_camera']) for name,v in params['methods']['ours']['views'].items()}
    paths={Path(spec['source_scene']),BASE/'case00/report.json',BASE/'case00/camera_params.json'}
    for method in lr.METHODS:
        paths.update([Path(previous['methods'][method]['final_scene']),lr.index_path(row,method)])
    before={str(p):r.sha(p) for p in paths}
    original=r.c.GaussianData.load(spec['source_scene'])
    report=dict(case='case00',revision='case00_ours_view_v2',annotation_red_opacity=.6,
        annotation_original_color_weight=.4,physical_gaussian_opacity_unchanged=True,
        full_native_localization_indices=True,all_methods_share_ours_cameras=True,
        training_steps=0,ply_writes=0,methods={},input_sha256=before,
        common_crop=previous['methods']['ours']['shared_crop'])
    cameras=dict(case='case00',reference=str(BASE/'case00/camera_params.json'),
        policy='Use previous Ours camera exactly for every method and stage, for main and both auxiliary views.',methods={})
    for method in lr.METHODS:
        ids=np.load(lr.index_path(row,method));idx=torch.as_tensor(ids,device='cuda')
        assert len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(original.xyz)
        scene=r.c.GaussianData.load(previous['methods'][method]['final_scene'])
        flags=torch.zeros_like(original.xyz);flags[idx]=1
        method_report=dict(localization_indices=str(lr.index_path(row,method)),
            selected_gaussians=len(ids),final_scene=previous['methods'][method]['final_scene'],views={})
        cameras['methods'][method]={}
        for name,cam in views.items():
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
            r.c.save_image(folder/'original.png',source)
            for suffix,image in [('localization',localization),('final',final),('localization_mask',contribution[None])]:
                r.c.save_image(folder/(method+'_'+suffix+'.png'),image)
            if name=='main':
                for suffix in ['localization','final']:
                    shutil.copy2(folder/(method+'_'+suffix+'.png'),OUT/(method+'_'+suffix+'.png'))
                shutil.copy2(folder/'original.png',OUT/'original.png')
            method_report['views'][name]=dict(nonselected_max_rgb_difference=error,
                visible_localization=r.alpha_stats(contribution.cpu().numpy()),
                exact_ours_camera=True,localization_final_camera_identical=True)
            cameras['methods'][method][name]=dict(localization_camera=cam.record(),final_camera=cam.record())
            print(method,name,'complete',flush=True)
        report['methods'][method]=method_report
        del scene,flags,color,source,localization,final,contribution
        torch.cuda.empty_cache()
    for p,h in before.items():assert r.sha(p)==h
    report['all_input_files_unchanged']=True
    r.save(OUT/'report.json',report);r.save(OUT/'camera_params.json',cameras)
    for stage in ['localization','final']:
        lr.board(OUT/('comparison_'+stage+'.png'),[[('Original',OUT/'original.png',None)]+
            [(label,OUT/(method+'_'+stage+'.png'),None) for method,label in zip(lr.METHODS,lr.LABELS)]],
            'case00 | '+stage+' | Exact common Ours view',CAPTION)
    for name,crop in [('comparison_paired',None),('comparison_crops',report['common_crop'])]:
        lr.board(OUT/(name+'.png'),[[(label+' / '+stage,OUT/(method+'_'+stage+'.png'),crop)
            for method,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],
            'case00 | Same-camera localization / final',CAPTION)
    for n in (1,2):
        lr.board(OUT/f'comparison_aux{n}.png',[[(label+' / '+stage,OUT/'views'/f'aux{n}'/(method+'_'+stage+'.png'),None)
            for method,label in zip(lr.METHODS,lr.LABELS)] for stage in ['localization','final']],
            f'case00 | Shared Ours auxiliary view {n}',CAPTION)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>case00：统一 Ours 视角 + 半透明定位</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:26px 0}</style><h1>case00：统一 Ours 视角 + 半透明定位</h1><p>GaussianEditor、DGE、GSLE、Ours 完全共用 Ours 原有相机；上排定位与下排最终结果也使用同一相机。辅助视角同样全部复用 Ours 的对应相机。红色标注采用 60% 红色 + 40% 原始颜色，让设备纹理可见；这只是定位显示颜色混合，没有改变高斯本身的透明度或三维场景。</p><p>各方法的最终 PLY 保持不变，生成位置的偏差在共同相机下如实保留。本次只修改 case00 的展示，其他案例未重新渲染或修改。</p><p><a href="../index.html">全部案例</a> | <a href="../case00/index.html">case00 上一版</a> | <a href="camera_params.json">相机参数</a> | <a href="report.json">检查记录</a></p>'''
    for name,label in [('comparison_paired','定位 / 最终结果'),('comparison_localization','半透明红色定位'),
                       ('comparison_final','最终结果'),('comparison_crops','统一区域局部放大'),
                       ('comparison_aux1','共用辅助视角 1'),('comparison_aux2','共用辅助视角 2')]:
        page+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (OUT/'index.html').write_text(page+'</html>',encoding='utf-8')
    # Only the case00 section and case00 overview row change in the active gallery.
    # Old case00 images and old overviews remain available unchanged.
    page=(BASE/'index.html').read_text(encoding='utf-8')
    page=page.replace('href="case00/index.html"','href="case00_ours_view_v2/index.html"')
    page=page.replace('href="case00/comparison_paired.png"','href="case00_ours_view_v2/comparison_paired.png"')
    page=page.replace('src="case00/comparison_paired.png"','src="case00_ours_view_v2/comparison_paired.png"')
    page=page.replace('<h1>','<p><strong>case00 已更新：四种方法共用 Ours 相机，红色定位为 60% 不透明度。其余案例保持原样。</strong></p><h1>',1)
    for stage in ['localization','final']:
        im=Image.open(BASE/('overview_'+stage+'.png')).convert('RGB')
        case=Image.open(OUT/('comparison_'+stage+'.png'))
        im.paste(case.crop((0,80,2500,620)),(0,90))
        from PIL import ImageDraw,ImageFont
        ImageDraw.Draw(im).text((12,128),'case00',font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
        name='overview_'+stage+'_case00_v2.png';im.save(BASE/name,dpi=(300,300))
        page=page.replace('href="overview_'+stage+'.png"','href="'+name+'"')
    (BASE/'index.html').write_text(page,encoding='utf-8')
    # Standalone case00 archive has no external HTML dependencies.
    with zipfile.ZipFile(OUT/'case00_ours_view_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.json','.html'):
                if p.name=='index.html':
                    text=p.read_text(encoding='utf-8')
                    text=text.replace('<a href="../index.html">全部案例</a> | ','').replace('<a href="../case00/index.html">case00 上一版</a> | ','')
                    z.writestr('index.html',text)
                else:z.write(p,p.relative_to(OUT))
    print('COMPLETE case00 revision',flush=True)


if __name__=='__main__':main()
