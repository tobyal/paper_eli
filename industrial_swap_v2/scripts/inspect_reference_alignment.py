"""Same-camera old/new placement and post-M4 comparison; no model updates."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
from industrial_swap_v2.core.pbr_gaussian_optimizer import render
from industrial_swap_v2.core.visibility_loss import object_contribution


def run(ctx):
    c=gsle(ctx);out=ctx.stage('m4');folder=out/'alignment';report=read_json(folder/'report.json')
    source=read_json(ctx.out/'provenance.json');old_dir=Path(source['previous_m4'])
    old=c.GaussianData.load(old_dir/'object_world.ply');new=c.GaussianData.load(out/'object_world.ply')
    bg=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    anchor=camera.resize(720,round(camera.height*720/camera.width))
    frame=report['space']['frame'];axes=np.asarray(frame['axes']);origin=np.asarray(frame['center'])
    box_low=np.asarray(frame['local_bbox_low']);box_high=np.asarray(frame['local_bbox_high'])
    pose_metrics={}
    for label,data,path in [('old',old,old_dir/'transform.json'),('new',new,out/'transform.json')]:
        xyz=data.xyz.cpu().numpy();local=(xyz-origin)@axes;low,high=np.quantile(local,[.002,.998],axis=0)
        t=read_json(path);rotation=np.asarray(t['rotation']);relative=axes.T@rotation
        mid=(low+high)/2;floor=mid.copy();floor[1]=low[1]
        pose_metrics[label]={'device_center_world':(origin+axes@mid).tolist(),'center_local':mid.tolist(),
            'scale':t['scale'],'yaw_degrees':float(np.degrees(np.arctan2(relative[0,2],relative[0,0])))%360,
            'center_bounds_low':low.tolist(),'center_bounds_high':high.tolist(),
            'center_bottom_gap_to_support':float(low[1]-box_low[1]),
            'center_bbox_overflow_fraction':((np.maximum(box_low-low,0)+np.maximum(high-box_high,0))/(box_high-box_low)).tolist(),
            'note':'Opacity-independent center quantiles for identical old/new geometry; covariance support metrics are in registration report'}
    def pil(t):return Image.fromarray(np.round(t.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype('uint8'))
    def board(images,labels,path):
        w=max(x.width for x in images);h=max(x.height for x in images)
        b=Image.new('RGB',(w*len(images),h+38),'white');d=ImageDraw.Draw(b)
        for j,(im,label) in enumerate(zip(images,labels)):
            b.paste(ImageOps.contain(im,(w,h)),(j*w,38));d.text((j*w+8,12),label,fill='black')
        b.save(path,quality=94)
    with torch.no_grad():
        ref=Image.open(ctx.stage('m2')/'object_white.png').convert('RGB').resize((anchor.width,anchor.height))
        before=pil(render(c,anchor,old)['rgb']);after=pil(render(c,anchor,new)['rgb'])
        board([ref,before,after],['Qwen reference',f'Old pose / IoU {report["baseline"]["mask_iou"]:.3f}',
            f'New pose / IoU {report["selected"]["mask_iou"]:.3f}'],folder/'reference_old_new.jpg')
        a=pil(c.render_joint(anchor,bg,old,aux=False)['rgb']);b=pil(c.render_joint(anchor,bg,new,aux=False)['rgb'])
        board([a,b],['Same original M1 background / old placement','Same original M1 background / new placement'],folder/'anchor_scene_old_new.jpg')
        c.save_image(folder/'anchor_scene_new.png',c.render_joint(camera,bg,new,aux=False)['rgb'])
        center=np.asarray(report['selected']['device_center_world']);views=[];directions=[]
        cameras=sorted(c.source_cameras(720),key=lambda cam:np.linalg.norm(cam.center-camera.center))
        for cam in cameras:
            uv,z=cam.project(center[None]);direction=cam.center-center;direction/=np.linalg.norm(direction)
            if z[0]<=0 or not(0<uv[0,0]<cam.width and 0<uv[0,1]<cam.height):continue
            if any(np.degrees(np.arccos(np.clip(direction@v,-1,1)))<7 for v in directions):continue
            standalone=c.raster(cam,new,torch.ones_like(new.xyz))[0]
            contribution=object_contribution(c,cam,bg,new)
            visible=float(contribution.sum()/standalone.sum().clamp_min(1))
            if visible<.35 or float(standalone.sum())<cam.width*cam.height*.002:continue
            composed=c.render_joint(cam,bg,new,aux=False)
            images=[pil(c.render_joint(cam,bg,old,aux=False)['rgb']),pil(composed['rgb'])]
            name=f'view_{len(views):02d}_{Path(cam.name).stem}.jpg'
            board(images,['Old placement','New placement'],folder/name)
            views.append({'name':cam.name,'camera':cam.record(),'comparison':name,'object_visible_fraction':visible});directions.append(direction)
            if len(views)==4:break
        if not views:raise ValueError('No calibrated inspection view sees the registered device')
        write_json(folder/'inspection_views.json',views)
    if (out/'summary.json').exists():
        summary=read_json(out/'summary.json')
        del old,bg;torch.cuda.empty_cache()
        initial=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
        actual=c.GaussianData.load(summary['candidate_background'])
        guarded=c.GaussianData.load(summary['guarded_background'])
        with torch.no_grad():
            for cam,name in [(anchor,'anchor'),*[(c.Camera.from_record(v['camera']),f'view_{i:02d}') for i,v in enumerate(views)]]:
                images=[pil(c.render_joint(cam,b,new,aux=False)['rgb']) for b in (initial,actual,guarded)]
                board(images,['New placement / initial',f'New placement / actual {summary["checks"]["training_steps"]}',
                    f'Environment-protected / step {summary["selected_step"]}'],folder/(name+'_m4_comparison.jpg'))
        report['m4']={k:summary[k] for k in ('candidate_guard_passed','selected_step','checks')}
    preserved={p:sha256(p)==expected for p,expected in read_json(ctx.out/'source_preservation.json').items()}
    if not all(preserved.values()):raise ValueError('Source experiment was changed')
    old_transform=read_json(old_dir/'transform.json');new_transform=read_json(out/'transform.json')
    a=np.asarray(old_transform['rotation']);b=np.asarray(new_transform['rotation'])
    report.update(source_preservation=preserved,other_views=views,pose_comparison=pose_metrics,
        old_transform=old_transform,new_transform=new_transform,
        scale_ratio=new_transform['scale']/old_transform['scale'],
        rotation_change_degrees=float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))))
    write_json(folder/'comparison_report.json',report)
    print('PLACEMENT_COMPARISON',report['baseline']['mask_iou'],report['selected']['mask_iou'],report['rotation_change_degrees'],flush=True)


if __name__=='__main__':
    import os
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();ctx=Context(a.config)
    os.environ.update(ctx.env());run(ctx)
