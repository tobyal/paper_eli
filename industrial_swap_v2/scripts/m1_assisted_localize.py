"""Human SAM reference, unchanged GSLE semantic/depth seed and box rules."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
import yaml
from ..core.context import Context, read_json, write_json, sha256


def region_from_target(c,target):
    up=np.array(target['up']);center=np.array(target['center']);ref=c.Camera.from_record(target['reference_camera'])
    toward=ref.center-center;front=toward-up*np.dot(toward,up);front/=np.linalg.norm(front)
    right=np.cross(up,front);right/=np.linalg.norm(right);axes=np.stack([right,up,front],1)
    local=np.array(target['bbox_corners'])@axes;low=local.min(0);high=local.max(0);sides=high-low
    high[1]=low[1]+max(sides)
    corners=c.bbox_corners(low,high)@axes.T
    return {**target,'bbox_corners':corners.tolist(),'center':(((low+high)/2)@axes.T).tolist(),
            'bbox_low':corners.min(0).tolist(),'bbox_high':corners.max(0).tolist(),
            'up':up.tolist(),'object_scale':float(max(high-low)),'region_axes':axes.tolist(),
            'original_bbox_corners':target['bbox_corners'],'original_oriented_side_lengths':sides.tolist(),
            'generation_oriented_side_lengths':(high-low).tolist(),'manual_coordinates':False,
            'generation_region_rule':'original GSLE: preserve footprint and bottom; height=max original extent'}


def inspection_cameras(cameras,reference,center):
    """Two registered observations of the same target, separated from reference."""
    reference_direction=reference.center-center;reference_direction/=np.linalg.norm(reference_direction)
    candidates=[]
    for camera in cameras:
        if camera.name==reference.name:continue
        xy,z=camera.project(center[None]);x,y=xy[0]
        if z[0]<=.01 or not 0<=x<camera.width or not 0<=y<camera.height:continue
        direction=camera.center-center;direction/=np.linalg.norm(direction)
        angle=float(np.degrees(np.arccos(np.clip(np.dot(direction,reference_direction),-1,1))))
        if 8<=angle<=65:candidates.append((abs(angle-25),camera,direction))
    chosen=[];directions=[]
    for _,camera,direction in sorted(candidates,key=lambda r:r[0]):
        if all(np.dot(direction,d)<np.cos(np.radians(12)) for d in directions):
            chosen.append(camera);directions.append(direction)
        if len(chosen)==2:break
    if len(chosen)<2:raise ValueError('Fewer than two distinct registered inspection views; choose another reference')
    return [reference]+chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);a=parser.parse_args()
    job=read_json(a.job);folder=a.job.parent;root=Path(job['root']);selection=Path(job['selection'])
    cfg_path=folder/'config.yaml';ctx=Context(cfg_path)
    os.environ.update(ROOT=str(root),GSLE_OUTPUT=str(folder/'adapter'),GSLE_RECORDS=str(folder/'records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(root/'gsle'));import common as c
    from world_up import estimate_up
    spec=read_json(selection/'selection.json');reference=np.asarray(Image.open(selection/'reference_mask.png'))>127
    cam=c.Camera.from_record(spec['camera']);fields=[np.load(folder/'semantic'/f'scores_level_{i}.npy',mmap_mode='r') for i in (1,2,3)]
    source_path=ctx.source(ctx.cfg['project']['source_ply'])
    if sha256(source_path)!=ctx.cfg['project']['source_sha256']:raise ValueError('Source Gaussian hash changed')
    source=c.GaussianData.load(source_path);xyz=source.xyz.cpu().numpy();scales=source.log_scale.exp().cpu().numpy()
    opacity=source.opacity_logit.sigmoid().cpu().numpy()[:,0];semantic=fields[spec['level']-1][:,0]
    floor_scores=np.max(np.stack([f[:,1] for f in fields]),axis=0)
    cameras=c.source_cameras(938);up,up_record=estimate_up(xyz,scales,opacity,floor_scores,cameras)
    xy,z=cam.project(xyz);pixels=np.round(xy).astype(np.int64)
    inside=(z>.01)&(pixels[:,0]>=0)&(pixels[:,0]<cam.width)&(pixels[:,1]>=0)&(pixels[:,1]<cam.height)
    sx=np.clip(pixels[:,0],0,cam.width-1);sy=np.clip(pixels[:,1],0,cam.height-1)
    with torch.no_grad():rendered=c.render(cam,source)
    distance=np.linalg.norm(xyz-cam.center,axis=-1)
    depth=(rendered['depth']/rendered['alpha'].clamp_min(1e-6))[0].cpu().numpy();local=depth[sy,sx]
    tolerance=np.maximum(scales.max(1)*3,.02*np.median(local[inside]))
    visible=inside&(abs(distance-local)<tolerance)
    seed=visible&reference[sy,sx]&(semantic>.5)&(opacity>.05)
    if seed.sum()<=100:raise ValueError(f'Only {seed.sum()} semantic/depth seeds; choose another description or SAM region. Thresholds were not relaxed.')
    low,high=np.quantile(xyz[seed],[.005,.995],axis=0);margin=(high-low)*.05;low-=margin;high+=margin
    box=((xyz>=low)&(xyz<=high)).all(1);preserved=0
    if 'offset' in up_record:
        floor=(abs(xyz@up+up_record['offset'])<up_record['tolerance'])&(floor_scores>semantic)
        preserved=int((box&floor).sum());box&=~floor
    ids=np.flatnonzero(box);keep=np.flatnonzero(~box)
    if not 100<len(ids)<len(xyz)*.2:raise ValueError('Unreasonable target extent; choose another description/mask. No case saved.')
    loc=folder/'localization';loc.mkdir(exist_ok=True)
    target={'text':job['text'],'reference_camera':cam.record(),'selected_level':spec['level'],'selected_sam_ids':spec['ids'],
            'seed_count':int(seed.sum()),'target_gaussians':len(ids),'retained_gaussians':len(keep),
            'bbox_low':low.tolist(),'bbox_high':high.tolist(),'bbox_corners':c.bbox_corners(low,high).tolist(),
            'center':((low+high)/2).tolist(),'up':up.tolist(),'up_estimation':up_record,'source_ply':str(source_path),
            'source_indices_preserved':True,'manual_roi_used':False,'human_sam_reference':True,'fresh_semantic_query':True,
            'extent_rule':'0.5%/99.5% visible text-positive seed quantiles plus 5% margin',
            'removal_rule':'centers inside semantic-derived box, preserving coplanar floor-semantic Gaussians',
            'preserved_floor_gaussians':preserved,'semantic_boundary':.5}
    write_json(loc/'target.json',target);write_json(loc/'edit_region.json',region_from_target(c,target))
    np.save(loc/'target_indices.npy',ids);np.save(loc/'keep_indices.npy',keep);np.save(loc/'seed_indices.npy',np.flatnonzero(seed))
    from ..modules.m1_localization import run
    run(ctx)
    out=ctx.stage('m1');manifest=read_json(out/'manifest.json');manifest.update(fresh_semantic_query=True,human_sam_reference=True,selection=str(selection),semantic_text=job['text'])
    write_json(out/'manifest.json',manifest)
    ctx.complete('m1',[out/n for n in ('manifest.json','target_gaussians.ply','background_gaussians.ply','target_indices.npy','keep_indices.npy')],counts=manifest['counts'])
    inspection=folder/'inspection';inspection.mkdir(exist_ok=True);views=[]
    flags=torch.zeros((len(xyz),3),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
    target_scene=source.subset(ids);background=source.subset(keep)
    for i,camera in enumerate(inspection_cameras(cameras,cam,(low+high)/2)):
        with torch.no_grad():
            full=c.render(camera,source);contribution=c.raster(camera,source,flags)[0].cpu().numpy()
            isolated=c.render(camera,target_scene);removed=c.render(camera,background)
        rgb=full['rgb'].permute(1,2,0).cpu().numpy().clip(0,1);mask=contribution>.05
        overlay=rgb.copy();overlay[mask]=.4*overlay[mask]+.6*np.array([1,.12,.12])
        alone_rgb=isolated['rgb'].permute(1,2,0).cpu().numpy();alpha=isolated['alpha'][0].cpu().numpy()
        images={'highlight':overlay,'target':alone_rgb+(1-alpha[:,:,None]),'background':removed['rgb'].permute(1,2,0).cpu().numpy()}
        for kind,image in images.items():Image.fromarray((image.clip(0,1)*255).astype(np.uint8)).save(inspection/f'view_{i}_{kind}.jpg',quality=92)
        Image.fromarray(mask.astype(np.uint8)*255).save(inspection/f'view_{i}_mask.png')
        views.append({'index':i,'camera':camera.record(),'target_visible_pixels':int(mask.sum())})
        if i==0:
            alone=alpha>.3
            quality={'reference_sam_iou':float((alone&reference).sum()/max((alone|reference).sum(),1)),
                     'projection_spill':float((mask&~reference).sum()/max(mask.sum(),1)),
                     'reference_recall':float((mask&reference).sum()/max(reference.sum(),1)),
                     'note':'SAM agreement is a diagnostic, not human ground truth; final decision is manual.'}
    result={'text':job['text'],'selection_id':spec['id'],'counts':manifest['counts'],'seed_count':int(seed.sum()),
            'views':views,'quality':quality,'human_confirmation_required':True,'qwen_calls':0,'trellis_calls':0,'executed_stages':['m1']}
    write_json(folder/'result.json',result)
    # Hash all immutable case inputs and outputs once for subsequent verification.
    integrity={str(p.relative_to(folder)):sha256(p) for branch in ('reference','localization','outputs','inspection','semantic')
               for p in (folder/branch).rglob('*') if p.is_file()}
    write_json(folder/'integrity.json',integrity);write_json(folder/'success.json',{'m1_only':True})
    print('M1 inspection ready; awaiting human confirmation',flush=True)


if __name__=='__main__':main()
