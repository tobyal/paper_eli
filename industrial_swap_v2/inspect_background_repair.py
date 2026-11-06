"""Inspect local floor geometry and attribute base occlusion to background Gaussians."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .core.context import Context, gsle, read_json, write_json
from .core.scene import read_vertices


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); ctx=Context(args.config); out=ctx.path(args.output); out.mkdir(parents=True,exist_ok=False)
    c=gsle(ctx); m=read_json(ctx.stage('m1')/'manifest.json'); region=read_json(ctx.stage('m1')/'edit_region.json')
    axis=np.asarray(m['world_transform']['axes']); center=np.asarray(m['region_center'])
    vertices=read_vertices(ctx.stage('m1')/'background_gaussians.ply')
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')]); local=(xyz-center)@axis
    scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]))
    quat=np.column_stack([vertices[f'rot_{i}'] for i in range(4)])[:,[1,2,3,0]]
    matrices=Rotation.from_quat(quat).as_matrix()
    shortest=matrices[np.arange(len(xyz)),:,np.argmin(scales,axis=1)]
    flat_alignment=np.abs(shortest@axis[:,1])
    extent=np.asarray(m['world_transform']['local_bbox_high'])-np.asarray(m['world_transform']['local_bbox_low'])
    nearby=(abs(local[:,0])<extent[0])&(abs(local[:,2])<extent[2])&(local[:,1]<0)
    semantic=np.zeros(len(xyz),dtype=np.float32)
    keep=np.load(ctx.stage('m1')/'keep_indices.npy')
    for level in (1,2,3):
        q=np.load(ctx.source(f'outputs/gsle/localization/language_query_level{level}.npz'))
        semantic=np.maximum(semantic,q['floor_scores'][keep])
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    bg=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    obj=c.GaussianData.load(ctx.stage('m3')/'object_world.ply')
    colors=bg.colors(camera.center).detach().cpu().numpy()
    luminance=colors@np.array([.2126,.7152,.0722])
    obj_local=(obj.xyz.detach().cpu().numpy()-center)@axis
    base=obj_local[:,1]<obj_local[:,1].min()+.15*np.ptp(obj_local[:,1])
    bg.opacity_logit=bg.opacity_logit.detach().requires_grad_(True)
    joint=c.concatenate(bg,obj)
    flags=torch.cat((torch.zeros((len(xyz),3),device='cuda'),torch.tensor(base,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)))
    rendered=c.raster(camera,joint,flags)
    rendered[0].sum().backward()
    gradient=-bg.opacity_logit.grad[:,0].detach().cpu().numpy()
    opacity=bg.opacity_logit.detach().sigmoid()[:,0].cpu().numpy()
    score=np.maximum(gradient,0)
    order=np.argsort(score)[::-1]
    c.save_image(out/'base_visible_before.png',rendered[:1])
    top=order[:3000]; torch.cuda.empty_cache()
    def summary(ids):
        if not len(ids): return {'count':0}
        return {'count':len(ids),'local_y_quantiles':np.quantile(local[ids,1],[0,.1,.25,.5,.75,.9,1]).tolist(),
            'max_sigma_quantiles':np.quantile(scales[ids].max(1),[0,.5,.9,.99,1]).tolist(),
            'floor_score_quantiles':np.quantile(semantic[ids],[0,.25,.5,.75,1]).tolist(),
            'normal_alignment_quantiles':np.quantile(flat_alignment[ids],[0,.25,.5,.75,1]).tolist(),
            'rgb_mean':np.mean(colors[ids],axis=0).tolist(),
            'local_bbox':[local[ids].min(0).tolist(),local[ids].max(0).tolist()]}
    groups={
        'nearby':np.flatnonzero(nearby),
        'nearby_semantic_floor':np.flatnonzero(nearby&(semantic>.55)),
        'nearby_flat_floor':np.flatnonzero(nearby&(semantic>.5)&(flat_alignment>.95)&(scales.max(1)<.06)&(opacity>.2)),
        'occluders_top100':order[:100],'occluders_top1000':order[:1000],
    }
    cumulative=np.cumsum(score[order]); fraction=cumulative/max(cumulative[-1],1e-8)
    report={'groups':{k:summary(v) for k,v in groups.items()},
        'old_floor_y':m['world_transform']['local_bbox_low'][1],
        'object_base_y':float(obj_local[:,1].min()),
        'occlusion_gradient_mass':float(score.sum()),
        'occluder_counts_for_gradient_mass':{str(p):int(np.searchsorted(fraction,p)+1) for p in (.5,.9,.95,.99)},
        'top_occluders':[{'index':int(i),'score':float(score[i]),'position':local[i].tolist(),'scale':scales[i].tolist(),
            'rgb':colors[i].tolist(),'floor_score':float(semantic[i]),'normal_alignment':float(flat_alignment[i])} for i in order[:25]]}
    write_json(out/'inspection.json',report)
    np.savez_compressed(out/'attributes.npz',local=local,scales=scales,normal_alignment=flat_alignment,floor_scores=semantic,
        colors=colors,opacity=opacity,occlusion_score=score,axes=axis,center=center)
    # Render the points responsible for most base occlusion, separately and omitted.
    n=int(np.searchsorted(fraction,.99)+1); selected=order[:n]
    all_indices=np.arange(len(xyz)); keep_mask=np.ones(len(xyz),bool); keep_mask[selected]=False
    bg.opacity_logit=bg.opacity_logit.detach()
    with torch.no_grad():
        c.save_image(out/'occluders_only.png',c.render(camera,bg.subset(torch.tensor(selected.copy(),device='cuda')),aux=False)['rgb'])
        c.save_image(out/'scene_without_top_occluders.png',c.render_joint(camera,bg.subset(torch.tensor(all_indices[keep_mask],device='cuda')),obj,aux=False)['rgb'])
    corners=(np.asarray(m['target_bbox'])-center)@axis
    hull=ConvexHull(corners[:,[0,2]])
    footprint=corners[hull.vertices][:,[0,2]]
    plane_y=-(center@axis[:,1]+region['up_estimation']['offset'])
    boundary=np.column_stack((footprint[:,0],np.full(len(footprint),plane_y),footprint[:,1]))@axis.T+center
    projected,_=camera.project(boundary)
    roi=np.zeros((camera.height,camera.width),np.uint8)
    cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(projected).astype(np.int32)),1)
    with torch.no_grad(): original_image=c.render(camera,bg,aux=False)['rgb']
    dark=(original_image.mean(0)<.30)&torch.tensor(roi>0,device='cuda')
    color_params=bg.colors(camera.center).detach().requires_grad_(True)
    attribution_image=c.raster(camera,bg,color_params)
    (attribution_image.sum(0)*dark).sum().backward()
    dark_score=color_params.grad[:,0].detach().cpu().numpy()
    c.save_image(out/'defect_mask.png',dark.float())
    np.savez_compressed(out/'dark_attribution.npz',score=dark_score,footprint=footprint,plane_y=plane_y)
    order2=np.argsort(dark_score)[::-1]
    report['footprint']=footprint.tolist(); report['footprint_projection']=projected.tolist()
    report['dark_pixels']=int(dark.sum()); report['top_dark']=[]
    for i in order2[:40]:
        report['top_dark'].append({'index':int(i),'score':float(dark_score[i]),'local':local[i].tolist(),
            'scales':scales[i].tolist(),'rgb':colors[i].tolist(),'alignment':float(flat_alignment[i])})
    write_json(out/'inspection.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
