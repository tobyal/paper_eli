"""Per-method object-centered cameras shared by localization and final renders.

Unlike render.py, no term aligns a baseline's viewing direction to Ours.
Own generation/localization cameras initialize framing. Ours is reusable only
when already geometrically close and comparably useful for the actual target.
All selection indices and source/final Gaussians remain unchanged.
"""
import argparse
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial.transform import Rotation

import render as r
import localization_red as lr

OUT=r.BASE/'target_centered_views'
METHODS=lr.METHODS
LABELS=lr.LABELS
CAPTION='Per-method target-centered views | Localization and final share one camera; methods may differ.'


def own_reference(row, method):
    folder=r.BASE/(method+'_native')/row['case']
    if method=='gaussianeditor':
        path=folder/'anchor_camera.json';record=r.read(path);key='Native Add camera'
    elif method=='dge':
        path=folder/'records/localization.json';record=r.read(path)['review_cameras'][0];key='Native primary localization camera'
    elif method=='gsle':
        path=folder/'avp_camera.json';record=r.read(path)['camera'];key='Native generation AVP camera'
    else:
        path=Path(row['ours_anchor']);record=r.read(path);key='Exact current Ours final anchor'
    cam=r.cpu_camera(record)
    assert cam.width==cam.height, 'Preserve aspect ratio'
    return cam.resize(1024,1024),dict(path=str(path),key=key,camera=record)


def aimed_camera(native, center, up, distance, yaw=0., pitch=0.):
    direction=r.c.normalized(native.center-center)
    q=Rotation.from_rotvec(up*np.radians(yaw)).as_matrix()
    right=r.c.normalized(np.cross(up,q@direction))
    q=Rotation.from_rotvec(right*np.radians(-pitch)).as_matrix()@q
    pos=center+distance*(q@direction)
    cam=r.c.look_at(pos,center,up,resolution=1024,name='own_target_centered')
    # Retain the method's native FOV, with target on the image-center ray.
    return r.c.Camera(cam.R,cam.t,1024,1024,native.fx,native.fy,512,512,cam.name)


def angle_between(a,b):
    return float(np.degrees(np.arccos(np.clip(r.c.normalized(a)@r.c.normalized(b),-1,1))))


@torch.no_grad()
def choose_camera(native, ours, center, up, info, obj, visibility_scene, flags,
                  original, localization_flags):
    trials=[]
    def evaluate(cam,kind,offset=(0.,0.)):
        m=r.candidate_metrics(cam,obj,visibility_scene,flags,res=224)
        low=cam.resize(224,224)
        local=r.c.raster(low,original,localization_flags)[0]
        m['localization_visible_pixels']=int((local>.02).sum())
        m['localization_visible_mass']=float(local.sum())
        m['center_ray_degrees']=angle_between(center-cam.center,cam.R[2])
        score=45*max(0,.985-m['capture'])+8*max(0,.80-m['visibility'])
        score+=1.6*abs(math.log(max(m['size'],.001)/.55))
        score+=2*max(0,.70-m['scene_coverage'])
        # Avoid a final-only view that conceals the original localization region.
        # Keep all native selection in renders, even disconnected false positives.
        score+=2*max(0,1-m['localization_visible_pixels']/180)
        score+=.05*(abs(offset[0])/15+abs(offset[1])/8)
        if kind=='native_exact':score-=.08
        item=dict(kind=kind,camera=cam.record(),score=score,yaw_offset=offset[0],pitch_offset=offset[1],**m)
        trials.append(item)
        return item
    evaluate(native,'native_exact')
    # Fit approximate support bounds first; actual rasterized alpha determines
    # the selected framing. Never recenter GSLE on diffusion outliers.
    bounds=r.corners(info['geometry'])
    distance=float(np.linalg.norm(native.center-center));lo=distance*.02;hi=distance*20
    for _ in range(28):
        mid=(lo+hi)/2;cam=aimed_camera(native,center,up,mid)
        xy,z=cam.project(bounds)
        size=np.ptp(xy,axis=0).max()/1024 if (z>0).all() else 1e6
        if size>.55:lo=mid
        else:hi=mid
    initial=(lo+hi)/2
    for factor in [.8,1.,1.35,1.8]:
        evaluate(aimed_camera(native,center,up,initial*factor),'own_target_centered')
    best=min(trials,key=lambda x:x['score'])
    if best['visibility']<.80 or best['capture']<.98 or best['localization_visible_pixels']<180:
        for yaw,pitch in [(-15,0),(15,0),(-30,0),(30,0),(0,8),(0,-8)]:
            evaluate(aimed_camera(native,center,up,initial,yaw,pitch),'own_target_offset',(yaw,pitch))
        best=min(trials,key=lambda x:x['score'])
    # Reuse Ours only if it already looks from essentially the native direction,
    # actually sees this target, and is not worse than independently chosen view.
    native_angle=angle_between(native.center-center,ours.center-center)
    bearing=angle_between(center-ours.center,ours.R[2])
    ours_record=dict(native_direction_difference_degrees=native_angle,
                     target_optical_axis_angle_degrees=bearing,reused=False)
    if native_angle<=18 and bearing<=22:
        candidate=evaluate(ours,'ours_exact_if_compatible')
        eligible=(candidate['capture']>=.98 and candidate['visibility']>=.65 and
                  .22<=candidate['size']<=.78 and candidate['localization_visible_pixels']>=180)
        ours_record['evaluated']=True;ours_record['eligible']=eligible
        if eligible and candidate['score']<=best['score']+.25:
            best=candidate;ours_record['reused']=True
    else:
        ours_record.update(evaluated=False,reason='Own native direction or target bearing is not close; no Ours-direction matching attempted.')
    return r.c.Camera.from_record(best['camera']),dict(selected=best,candidates=trials,
        ours_reuse=ours_record,initial_distance=initial,
        policy='Own target first; native generation direction; same camera for red localization and final. No cross-method direction-matching loss.')


def figure(out, rows, name, title):
    lr.board(out/(name+'.png'),rows,title,CAPTION)


@torch.no_grad()
def run(case):
    started=time.time();spec=r.read(r.BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']==case)
    out=OUT/case;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'report.json').exists(),'Preserve completed output'
    previous=r.read(r.BASE/'matched_views'/case/'camera_params.json')
    source_hash=r.sha(spec['source_scene']);original=r.c.GaussianData.load(spec['source_scene'])
    ours,ours_source=own_reference(row,'ours')
    up=r.c.normalized(np.array(r.read(row['target_record'])['up']))
    cameras=dict(case=case,policy=CAPTION,world_up=up.tolist(),methods={})
    report=dict(case=case,source_scene=spec['source_scene'],source_scene_sha256=source_hash,
        training_steps=0,ply_writes=0,source_files_unchanged=True,methods={},
        localization='ALL saved native indices, visualized red in pre-edit source scene, true opacity/depth; not projected bounding boxes.',
        final='Exact final PLY from previous comparison; original native DI-SDS for GSLE.',
        cross_method_note='Cameras may differ substantially across methods. Within each method localization and final use identical cameras.')
    for method in METHODS:
        info=dict(previous['methods'][method]['target'])
        if method=='dge':
            focus=r.BASE/'matched_views'/case/'dge_display_focus_indices.npy'
            info['indices']=np.load(focus).tolist()
        paths={Path(info['scene_path']),lr.index_path(row,method)}
        if info.get('object_path'):paths.add(Path(info['object_path']))
        native,native_record=own_reference(row,method);paths.add(Path(native_record['path']))
        before={str(p):r.sha(p) for p in paths}
        scene=r.c.GaussianData.load(info['scene_path'])
        obj,vis_scene,flags=r.flag_data(info,scene,original)
        ids=np.load(lr.index_path(row,method));assert len(ids)==len(np.unique(ids)) and ids.min()>=0 and ids.max()<len(original.xyz)
        idx=torch.as_tensor(ids,device='cuda');loc_flags=torch.zeros_like(original.xyz);loc_flags[idx]=1
        center=np.asarray(info['center'])
        if method=='ours':
            cam=ours;selection=dict(policy='Exact saved Ours final anchor',ours_reuse=dict(reused=True))
        else:
            cam,selection=choose_camera(native,ours,center,up,info,obj,vis_scene,flags,original,loc_flags)
        print(case,method,'selected',selection.get('selected',{}).get('kind','ours_exact'),
              selection.get('selected',{}).get('visibility'),flush=True)
        views=[('main',cam)]
        for k,angle in [(1,-20),(2,20)]:
            # Rotate each method's OWN camera rig; no Ours auxiliary template.
            aux=r.camera_at(cam,center,center,np.linalg.norm(cam.center-center),up,angle,0,method+f'_aux{k}')
            views.append((f'aux{k}',aux))
        method_report=dict(target_center=info['center'],center_source=info['center_source'],
            target_rule=info['center_rule'],final_scene=info['scene_path'],
            target_index_source=str(lr.index_path(row,method)),localized_gaussians=len(ids),
            native_reference=native_record,selection=selection,views={},input_sha256=before)
        camera_views={}
        for label,view in views:
            folder=out/'views'/label;folder.mkdir(parents=True,exist_ok=True)
            col=original.colors(view.center);base=r.c.raster(view,original,col)
            col[idx]=col.new_tensor([1.,0.,0.]);red=r.c.raster(view,original,col)
            final=r.c.render(view,scene,aux=False)['rgb'];contrib=r.c.raster(view,original,loc_flags)[0]
            outside=contrib==0
            diff=float((red-base).abs()[:,outside].max()) if outside.any() else 0.
            assert diff<2e-6,diff
            a=r.object_alpha(view,obj);m=r.alpha_stats(a)
            for suffix,image in [('original',base),('localization',red),('final',final),('localization_mask',contrib[None]),
                                 ('object_alpha',torch.from_numpy(a)[None])]:
                r.c.save_image(folder/(method+'_'+suffix+'.png'),image)
            camera_views[label]=dict(localization_camera=view.record(),final_camera=view.record(),original_camera=view.record())
            method_report['views'][label]=dict(camera_identical=True,object_projection=m,
                localization_projection=r.alpha_stats(contrib.cpu().numpy()),
                nonselected_max_pixel_change=diff,distance=float(np.linalg.norm(view.center-center)))
            if label=='main':
                # One shared crop per method, covering both final target and
                # visible localization near it; never independently align stages.
                box=m['bbox'];local_box=method_report['views'][label]['localization_projection']['bbox']
                if box is None:box=local_box
                elif local_box:
                    box=[min(box[0],local_box[0]),min(box[1],local_box[1]),max(box[2],local_box[2]),max(box[3],local_box[3])]
                method_report['shared_crop']=r.square_crop(box,1024)
                for suffix in ['original','localization','final']:
                    shutil.copy2(folder/(method+'_'+suffix+'.png'),out/(method+'_'+suffix+'.png'))
            print(case,method,label,'rendered',flush=True)
        assert all(r.sha(p)==value for p,value in before.items())
        method_report['source_files_unchanged']=True
        report['methods'][method]=method_report
        cameras['methods'][method]=dict(target_center=info['center'],native_reference=native_record,
            selection=selection,views=camera_views)
        del obj,scene,vis_scene,flags,loc_flags,col,base,red,final,contrib
        torch.cuda.empty_cache()
    assert r.sha(spec['source_scene'])==source_hash
    r.save(out/'camera_params.json',cameras)
    report['seconds']=time.time()-started;r.save(out/'report.json',report)
    make_figures(out,case,report)
    print('COMPLETE',case,report['seconds'],flush=True)


def make_figures(out,case,report):
    original=('Original',out/'ours_original.png',None)
    for suffix in ['localization','final']:
        figure(out,[[original]+[(l,out/(m+'_'+suffix+'.png'),None) for m,l in zip(METHODS,LABELS)]],
            'comparison_'+suffix,case+' | '+('Localized Gaussians in red' if suffix=='localization' else 'Final Gaussian scenes'))
    figure(out,[[(l+' / '+stage,out/(m+'_'+suffix+'.png'),None) for m,l in zip(METHODS,LABELS)]
        for suffix,stage in [('localization','Localization'),('final','Final')]],'comparison_paired',case+' | Same-camera localization / final')
    figure(out,[[(l+' / '+stage,out/(m+'_'+suffix+'.png'),report['methods'][m]['shared_crop']) for m,l in zip(METHODS,LABELS)]
        for suffix,stage in [('localization','Localization'),('final','Final')]],'comparison_crops',case+' | Identical crops within each method')
    for i in (1,2):
        figure(out,[[(l+' / '+stage,out/'views'/f'aux{i}'/(m+'_'+suffix+'.png'),None) for m,l in zip(METHODS,LABELS)]
            for suffix,stage in [('localization','Localization'),('final','Final')]],f'comparison_aux{i}',case+f' | Auxiliary {i}: paired cameras')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{case} 定位与生成视角对齐</title>
    <style>body{{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px;background:#f6f7f9}}section{{background:white;margin:24px 0;padding:16px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ddd}}</style>
    <h1>{case}：各方法面向自身目标，定位 / 最终结果使用同一相机</h1><p><a href="../index.html">全部案例</a> | <a href="camera_params.json">相机参数</a> | <a href="report.json">来源和检查记录</a></p>
    <p>优先采用各方法自身生成 / 定位参考方向，以其实际目标为中心取景，不再向 Ours 方向靠拢。只有方向接近且实际可见性、完整度合适，才复用 Ours 相机。上下两排使用完全一致的相机，不移动任何 Gaussian。</p>
    <p>红色是原场景中各方法保存的全部定位高斯；下排是真实最终 PLY。DGE 取景中心采用其原生主视角贡献最大的定位连通区域，红色显示仍保留全部选择。GSLE 使用原始 Native DI-SDS 最终结果。生成位置与原定位不重合时，在同相机下如实呈现。</p><table><tr><th>方法</th><th>选用视角</th><th>复用 Ours</th></tr>'''
    for m,l in zip(METHODS,LABELS):
        s=report['methods'][m]['selection'];page+=f'<tr><td>{l}</td><td>{s.get("selected",{}).get("kind","ours_exact")}</td><td>{s["ours_reuse"]["reused"]}</td></tr>'
    page+='</table>'
    for name,label in [('comparison_paired','定位 / 最终结果成对比较'),('comparison_final','最终场景'),
                       ('comparison_localization','红色定位'),('comparison_crops','同区域局部放大'),
                       ('comparison_aux1','辅助视角 1'),('comparison_aux2','辅助视角 2')]:
        page+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (out/'index.html').write_text(page+'</html>',encoding='utf-8')


def package():
    for kind in ['final','localization']:
        canvas=Image.new('RGB',(2500,90+540*len(lr.CASES)),'white');d=ImageDraw.Draw(canvas)
        d.text((16,12),'Per-method target-centered views | '+kind,font=ImageFont.truetype('DejaVuSans.ttf',27),fill='black')
        d.text((16,50),CAPTION,font=ImageFont.truetype('DejaVuSans.ttf',22),fill='#444444')
        for i,case in enumerate(lr.CASES):
            im=Image.open(OUT/case/('comparison_'+kind+'.png'));canvas.paste(im.crop((0,80,2500,620)),(0,90+i*540))
            d.text((12,90+i*540+38),case,font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
        canvas.save(OUT/('overview_'+kind+'.png'),dpi=(300,300))
        ImageOps.contain(canvas,(1600,1900)).save(OUT/('overview_'+kind+'_web.jpg'),quality=94)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>定位与最终生成：同相机成对展示</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:30px 0}</style><h1>定位与最终生成：同相机成对展示</h1><p>每个方法朝向自己实际生成 / 定位的目标，不再优先贴近 Ours 方向。每个方法的红色定位与最终结果使用完全相同的相机。不同方法的观察方向和相机位置可以不同。仅在视角接近且能完整展示目标时复用 Ours 相机。</p><p>只读取已有 PLY 渲染，没有训练、重新定位或修改场景。GSLE 使用原始 Native DI-SDS。</p>'''
    for case in lr.CASES:
        page+=f'<section><h2><a href="{case}/index.html">{case} 完整结果</a></h2><p>上排：原始场景红色定位；下排：最终生成场景。每列共用一台相机。</p><a href="{case}/comparison_paired.png"><img loading="lazy" src="{case}/comparison_paired.png"></a></section>'
    page+='<p><a href="overview_final.png">最终结果总览</a> | <a href="overview_localization.png">对应定位总览</a></p></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    reports=[r.read(OUT/case/'report.json') for case in lr.CASES]
    assert all(x['source_files_unchanged'] for x in reports)
    r.save(OUT/'validation.json',dict(cases=lr.CASES,training_steps=0,ply_writes=0,all_sources_unchanged=True,
        paired_cameras_identical=all(v['camera_identical'] for x in reports for m in x['methods'].values() for v in m['views'].values())))
    (OUT/'figure_caption.txt').write_text('Per-method target-centered rendering. Each method is viewed around its own actual generated or localized target, prioritizing its native generation/reference direction rather than matching Ours. Within each method, pre-edit red localization and the final Gaussian scene are rendered with identical camera intrinsics and extrinsics. Cross-method camera positions and orientations may differ. Ours cameras are reused only when compatible with the actual target and native direction. All saved localization indices, including false positives, are retained. No Gaussian PLY or experiment output is changed. GSLE uses its original native DI-SDS final scene. DGE framing uses its own native-primary-visible connected component, without discarding any selection from localization displays.')
    with zipfile.ZipFile(OUT/'target_centered_pairs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.jpg','.json','.html','.txt'):z.write(p,p.relative_to(OUT))
    print('PACKAGED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--package',action='store_true');a=p.parse_args()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45);OUT.mkdir(parents=True,exist_ok=True)
    if a.case:run(a.case)
    elif not a.package:
        for case in lr.CASES:run(case)
    if a.package or not a.case:package()
