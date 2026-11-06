"""Object-Aware Gaussian Occlusion Filtering (OAGOF).

Conservative, deletion-only filtering. No floor, semantic color, or generation
model dependency. Coordinates are world-space COLMAP; camera[0] is the anchor.
The renderer is the project's existing Graphdeco differentiable rasterizer.
"""
from dataclasses import dataclass, asdict
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from .context import read_json, write_json, sha256


@dataclass
class FilterConfig:
    enabled: bool = False
    use_multi_view: bool = True
    preserve_object_geometry: bool = True
    floor_repair: bool = False
    resolution: int = 640
    max_views: int = 6
    max_guard_views: int = 8
    min_confirm_views: int = 2
    min_view_separation_degrees: float = 8.0
    support_sigma: float = 3.0
    region_expansion: float = .08
    object_proximity: float = .06
    min_occlusion_fraction: float = .000015
    min_inside_ratio: float = .25
    depth_margin_fraction: float = .002
    max_delete_fraction: float = .002
    max_outside_mae: float = .0015
    max_outside_changed_fraction: float = .003
    outside_change_threshold: float = .03
    mask_alpha_threshold: float = .25
    min_reference_iou: float = .35


def filter_config(value=None):
    if value is not None and not isinstance(value,dict): raise ValueError('OAGOF config must be an object')
    unknown=set(value or {})-set(FilterConfig.__dataclass_fields__)
    if unknown: raise ValueError('Unknown OAGOF parameters: '+', '.join(sorted(unknown)))
    cfg=FilterConfig(**(value or {}))
    for key in ('enabled','use_multi_view','preserve_object_geometry','floor_repair'):
        if type(getattr(cfg,key)) is not bool: raise ValueError(key+' must be boolean')
    if not cfg.preserve_object_geometry or cfg.floor_repair:
        raise ValueError('OAGOF requires immutable object geometry and floor_repair=false')
    for key in ('resolution','max_views','max_guard_views','min_confirm_views'):
        if type(getattr(cfg,key)) is not int: raise ValueError(key+' must be integer')
    if not 128<=cfg.resolution<=1280 or not 1<=cfg.max_views<=16 or not 1<=cfg.min_confirm_views<=cfg.max_views:
        raise ValueError('Invalid OAGOF resolution/view count')
    if not 0<=cfg.max_guard_views<=16: raise ValueError('Invalid OAGOF guard view count')
    for key in set(asdict(cfg))-{'enabled','use_multi_view','preserve_object_geometry','floor_repair','resolution','max_views','max_guard_views','min_confirm_views'}:
        value=getattr(cfg,key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(key+' must be finite and positive')
    if not 1<=cfg.support_sigma<=5 or cfg.max_delete_fraction>.01:
        raise ValueError('Support sigma must be 1–5; deletion cap must be <= 1%')
    if cfg.region_expansion>.5 or cfg.object_proximity>.5 or cfg.min_inside_ratio>1 or cfg.min_reference_iou>1:
        raise ValueError('Invalid relative geometry/mask thresholds')
    if cfg.max_outside_mae>.05 or cfg.max_outside_changed_fraction>.05:
        raise ValueError('Outside-mask guards must be <= 5%')
    return cfg


def covariance(vertices):
    from scipy.spatial.transform import Rotation
    scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]))
    q=np.column_stack([vertices[f'rot_{i}'] for i in (1,2,3,0)])
    rotation=Rotation.from_quat(q).as_matrix()
    return scales,rotation


def spatial_context_digest(edit_region,cameras):
    frame=edit_region.get('world_transform',edit_region)
    payload={'frame':frame,'cameras':[cam.record() if hasattr(cam,'record') else cam for cam in cameras]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def spatial_candidates(vertices, object_xyz, edit_region, cfg):
    """Ellipsoid support tests; an outside center alone does not exclude it."""
    from scipy.spatial import cKDTree
    frame=edit_region.get('world_transform',edit_region)
    axes=np.asarray(frame['axes']); center=np.asarray(frame['center'])
    low=np.asarray(frame['local_bbox_low']); high=np.asarray(frame['local_bbox_high'])
    if not np.allclose(axes.T@axes,np.eye(3),atol=1e-4) or np.any(high<=low):
        raise ValueError('Invalid M1 oriented editing region')
    size=float(np.linalg.norm(high-low)); expansion=cfg.region_expansion*size
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')]); local=(xyz-center)@axes
    scales,rotation=covariance(vertices)
    # Check an actual point in the expanded box against each covariance.
    # This is conservative for oblique ellipsoids: uncertain intersections stay.
    closest=np.clip(local,low-expansion,high+expansion)@axes.T+center
    delta=np.einsum('nji,nj->ni',rotation,closest-xyz)
    box_support=np.sum((delta/np.maximum(scales,1e-12))**2,axis=1)<=cfg.support_sigma**2
    tree=cKDTree(object_xyz[::max(1,len(object_xyz)//40000)])
    distance,nearest=tree.query(xyz,workers=4)
    probe=tree.data[nearest]; delta=np.einsum('nji,nj->ni',rotation,probe-xyz)
    object_support=np.sum((delta/(scales+cfg.object_proximity*size))**2,axis=1)<=cfg.support_sigma**2
    # Both old editing space and proximity to the new object are required.
    # Prevent a world-sized covariance from authorizing unrelated foreground.
    candidate=box_support&object_support&(scales.max(1)<size)&(distance<2*size)
    return np.flatnonzero(candidate),xyz,scales,rotation,size


def select_cameras(cameras, object_xyz, cfg):
    """Original calibrated cameras, with distinct viewing directions."""
    anchor=cameras[0]; center=np.median(object_xyz,axis=0)
    probe=object_xyz[::max(1,len(object_xyz)//2000)]
    direction=lambda cam:(cam.center-center)/np.linalg.norm(cam.center-center)
    first=direction(anchor); pool=[]
    for cam in cameras[1:]:
        uv,z=cam.project(probe)
        visible=(z>0)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
        angle=np.degrees(np.arccos(np.clip(direction(cam)@first,-1,1)))
        if visible.mean()>.9 and angle<65 and np.ptp(uv[visible],axis=0).prod()/(cam.width*cam.height)>.003:
            pool.append(cam)
    selected=[anchor]
    if cfg.use_multi_view:
        while pool and len(selected)<cfg.max_views:
            distances=np.degrees(np.arccos(np.clip(np.array([direction(c) for c in pool])@np.array([direction(c) for c in selected]).T,-1,1))).min(1)
            # Prefer useful nearby views while still separating all votes.
            index=int(np.argmax(np.minimum(distances,20)))
            if distances[index]<cfg.min_view_separation_degrees: break
            selected.append(pool.pop(index))
    return [cam.resize(cfg.resolution,2*round(cam.height*cfg.resolution/cam.width/2)) for cam in selected]


def select_deletions(gain, inside, outside, front, object_mass, outside_pixels, cfg, background_count):
    """Relative contribution, multi-view votes, and conservative outside budget."""
    normalized=gain/np.maximum(object_mass[:,None],1)
    ratio=inside/np.maximum(inside+outside,1e-8)
    votes=(normalized>=cfg.min_occlusion_fraction)&(ratio>=cfg.min_inside_ratio)&front
    required=cfg.min_confirm_views if cfg.use_multi_view else 1
    eligible=votes.sum(0)>=required
    score=np.where(votes,normalized,0).sum(0)
    cost=(outside/np.maximum(outside_pixels[:,None],1)).sum(0)
    order=np.argsort(-(score/(cost+1e-5)),kind='stable')
    chosen=[]; spent=np.zeros(len(gain)); cap=int(background_count*cfg.max_delete_fraction)
    for index in order:
        if not eligible[index] or len(chosen)>=cap: continue
        proposed=spent+outside[:,index]/np.maximum(outside_pixels,1)
        if np.any(proposed>cfg.max_outside_mae*.75): continue
        chosen.append(int(index)); spent=proposed
    return np.array(chosen,dtype=np.int64),votes,score


def select_guard_cameras(cameras, object_xyz, votes, cfg):
    """Cover other sides of the object for collateral checks, not delete votes."""
    if not cfg.use_multi_view: return []
    center=np.median(object_xyz,axis=0); probe=object_xyz[::max(1,len(object_xyz)//1000)]
    names={cam.name for cam in votes}; pool=[]
    direction=lambda cam:(cam.center-center)/np.linalg.norm(cam.center-center)
    for cam in cameras[1:]:
        if cam.name in names: continue
        uv,z=cam.project(probe)
        visible=(z>0)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
        if visible.mean()>.9 and np.ptp(uv[visible],axis=0).prod()/(cam.width*cam.height)>.003: pool.append(cam)
    guards=[]
    while pool and len(guards)<cfg.max_guard_views:
        angles=np.degrees(np.arccos(np.clip(np.array([direction(c) for c in pool])@np.array([direction(c) for c in votes+guards]).T,-1,1))).min(1)
        i=int(np.argmax(angles))
        if angles[i]<cfg.min_view_separation_degrees/2: break
        guards.append(pool.pop(i))
    return [cam.resize(cfg.resolution,2*round(cam.height*cfg.resolution/cam.width/2)) for cam in guards]


def filter_object_occlusion(background_path, object_path, edit_region, reference_mask,
                            cameras, output_dir, config=None):
    """Return output paths/report. Caller initializes gsle.common once.

    cameras: common.Camera objects or records, anchor first, then originals.
    reference_mask: M2 full-canvas mask path (only used for camera[0]).
    edit_region: M1 manifest or world_transform dict, never a floor estimate.
    Existing output directories are refused; original inputs are read-only.
    """
    import cv2
    import torch
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from .scene import read_vertices, save_vertices, compose
    root=Path(__file__).resolve().parents[2]
    os.environ.setdefault('ROOT',str(root)); sys.path.insert(0,str(root/'gsle'))
    import common as c
    cameras=[c.Camera.from_record(cam) if isinstance(cam,dict) else cam for cam in cameras]
    if not cameras: raise ValueError('At least the anchor camera is required')
    if isinstance(edit_region,(str,Path)): edit_region=read_json(edit_region)
    cfg=filter_config(config)
    if not cfg.enabled: raise ValueError('Enable OAGOF before invoking the filter')
    output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); background_path=Path(background_path); object_path=Path(object_path)
    inputs={'background':sha256(background_path),'object':sha256(object_path),'reference_mask':sha256(reference_mask)}
    write_json(output_dir/'effective_config.json',asdict(cfg))
    vertices=read_vertices(background_path); object_vertices=read_vertices(object_path)
    object_xyz=np.column_stack([object_vertices[k] for k in ('x','y','z')])
    if not len(vertices) or not len(object_vertices): raise ValueError('Background and object must be nonempty')
    ids,xyz,scales,rotation,size=spatial_candidates(vertices,object_xyz,edit_region,cfg)
    views=select_cameras(cameras,object_xyz,cfg); vote_count=len(views)
    views+=select_guard_cameras(cameras,object_xyz,views,cfg)
    write_json(output_dir/'cameras.json',[cam.record() for cam in views])
    bg=c.GaussianData.load(background_path); obj=c.GaussianData.load(object_path)
    alpha=bg.opacity_logit.detach().sigmoid()[:,0].cpu().numpy()
    gain=[]; inside=[]; outside=[]; front=[]; masses=[]; outside_counts=[]; frames=[]; view_reports=[]
    arrays=output_dir/'renders'; arrays.mkdir()
    for vi,cam in enumerate(views):
        print(f'OAGOF attribution {vi+1}/{len(views)} {cam.name}',flush=True)
        with torch.no_grad():
            zobj=(obj.xyz@torch.as_tensor(cam.R[2],device='cuda')+float(cam.t[2]))
            packed=c.raster(cam,obj,torch.stack((torch.ones_like(zobj),zobj,torch.zeros_like(zobj)),dim=1))
            object_alpha=packed[0]; depth=packed[1]/object_alpha.clamp_min(1e-6)
            isolated=c.render(cam,obj,aux=False)['rgb']; background=c.render(cam,bg,aux=False)['rgb']
            before=c.render_joint(cam,bg,obj,aux=False)['rgb']
        mask=object_alpha.detach().cpu().numpy()>cfg.mask_alpha_threshold
        if mask.sum()<128:
            view_reports.append({'camera':cam.name,'skipped':'insufficient visible object pixels'}); continue
        evidence=mask.copy(); iou=None
        is_guard=vi>=vote_count
        edge=max(1,round(cfg.resolution*.003))
        if vi==0:
            with Image.open(reference_mask) as im:
                reference=np.asarray(im.convert('L').resize((cam.width,cam.height),Image.Resampling.NEAREST))>127
            iou=float(np.count_nonzero(mask&reference)/max(1,np.count_nonzero(mask|reference)))
            dilated=cv2.dilate(reference.astype(np.uint8),np.ones((2*edge+1,2*edge+1),np.uint8)).astype(bool)
            evidence&=dilated
            if iou<cfg.min_reference_iou: evidence[:]=False
            c.save_image(arrays/f'{vi:02d}_reference_mask.png',reference.astype(float))
        guard=cv2.dilate(mask.astype(np.uint8),np.ones((2*edge+1,2*edge+1),np.uint8)).astype(bool)
        mt=torch.tensor(evidence,device='cuda',dtype=torch.float32)
        # Gradient of object visibility only: background behind the device has
        # zero effect. Divide sigmoid saturation without changing any alpha.
        bg.opacity_logit=bg.opacity_logit.detach().requires_grad_(not is_guard)
        joint=c.concatenate(bg,obj)
        flags=torch.cat((torch.zeros((len(vertices),3),device='cuda'),torch.ones((len(object_vertices),3),device='cuda')))
        visible=c.raster(cam,joint,flags)[0]
        if not is_guard:
            (visible*mt).sum().backward()
            derivative=-bg.opacity_logit.grad[:,0].detach().cpu().numpy()
            improvement=np.maximum(derivative,0)/np.maximum(1-alpha,.01)
        else: improvement=np.zeros(len(vertices),dtype=np.float32)
        bg.opacity_logit=bg.opacity_logit.detach()
        visible_before=visible.detach().cpu().numpy(); del visible,joint
        # Exact visible support inside/outside the object's own mask. This
        # protects nearby rails/devices even when their centers are nearby.
        color_parameters=torch.zeros((len(vertices),3),device='cuda',requires_grad=True)
        contribution=c.raster(cam,bg,color_parameters)
        (contribution[0]*torch.tensor(mask,device='cuda')+contribution[1]*torch.tensor(~guard,device='cuda')).sum().backward()
        weights=color_parameters.grad.detach().cpu().numpy(); del color_parameters,contribution
        # Project full anisotropic covariance, including off-center supports.
        uv,z=cam.project(xyz[ids]); cp=xyz[ids]@cam.R.T+cam.t
        covariance_world=np.einsum('nik,nk,njk->nij',rotation[ids],scales[ids]**2,rotation[ids])
        covcam=np.einsum('ij,njk,lk->nil',cam.R,covariance_world,cam.R)
        jac=np.zeros((len(ids),2,3)); safe_z=np.maximum(z,1e-6)
        jac[:,0,0]=cam.fx/safe_z; jac[:,0,2]=-cam.fx*cp[:,0]/safe_z**2
        jac[:,1,1]=cam.fy/safe_z; jac[:,1,2]=-cam.fy*cp[:,1]/safe_z**2
        cov2=np.einsum('nij,njk,nlk->nil',jac,covcam,jac)+np.eye(2)[None]*.3
        _,nearest=distance_transform_edt(~evidence,return_indices=True)
        pixel=np.rint(uv).astype(np.int64); clipped=pixel.copy()
        clipped[:,0]=np.clip(clipped[:,0],0,cam.width-1); clipped[:,1]=np.clip(clipped[:,1],0,cam.height-1)
        py,px=nearest[:,clipped[:,1],clipped[:,0]]
        delta=np.column_stack((px,py))-uv
        projection_overlap=np.einsum('ni,nij,nj->n',delta,np.linalg.inv(cov2),delta)<=cfg.support_sigma**2
        sampled=depth.detach().cpu().numpy()[py,px]
        foreground=(z-.5*np.sqrt(np.maximum(covcam[:,2,2],0))<sampled-cfg.depth_margin_fraction*size)&(z>0)
        foreground&=projection_overlap&bool(evidence.any())
        gain.append(improvement[ids]); inside.append(weights[ids,0]); outside.append(weights[ids,1]); front.append(foreground)
        masses.append(float((object_alpha*mt).sum())); outside_counts.append(int((~guard).sum()))
        frames.append((cam,mask,guard,before.detach().cpu(),object_alpha.detach().cpu(),visible_before))
        view_reports.append({'camera':cam.name,'role':'collateral_guard' if is_guard else 'occlusion_vote',
            'reference_iou':iou,'evidence_pixels':int(evidence.sum()),'object_mask_pixels':int(mask.sum())})
        c.save_image(arrays/f'{vi:02d}_object.png',isolated); c.save_image(arrays/f'{vi:02d}_background.png',background)
        c.save_image(arrays/f'{vi:02d}_object_mask.png',mask.astype(float)); c.save_image(arrays/f'{vi:02d}_evidence_mask.png',evidence.astype(float))
        c.save_image(arrays/f'{vi:02d}_depth_preview.png',depth.detach()/depth.max().clamp_min(1e-6))
        np.save(output_dir/f'{vi:02d}_object_depth.npy',depth.detach().cpu().numpy())
        del improvement,weights,flags,packed,isolated,background,before,object_alpha,depth
        torch.cuda.empty_cache()
    if not frames: raise ValueError('No usable object views; no background modified')
    gain=np.asarray(gain); inside=np.asarray(inside); outside=np.asarray(outside); front=np.asarray(front)
    selected,votes,score=select_deletions(gain,inside,outside,front,np.asarray(masses),np.asarray(outside_counts),cfg,len(vertices))
    np.savez_compressed(output_dir/'attribution.npz',background_indices=ids,occlusion_gain=gain,
        inside_contribution=inside,outside_contribution=outside,foreground_support=front,votes=votes,score=score)
    def evaluate(chosen,save=False):
        keep=np.ones(len(vertices),bool); keep[ids[chosen]]=False
        filtered=bg.subset(torch.tensor(np.flatnonzero(keep),device='cuda'))
        metrics=[]; accepted=True
        with torch.no_grad():
            joint=c.concatenate(filtered,obj)
            flags=torch.cat((torch.zeros((int(keep.sum()),3),device='cuda'),torch.ones((len(object_vertices),3),device='cuda')))
            for vi,(cam,mask,guard,before,isolated_alpha,visible_before) in enumerate(frames):
                after=c.render_joint(cam,filtered,obj,aux=False)['rgb'].cpu()
                visible_after=c.raster(cam,joint,flags)[0].cpu().numpy()
                diff=(after-before).abs().numpy(); changed=diff.max(0)>cfg.outside_change_threshold
                mass=float(isolated_alpha.numpy()[mask].sum())
                outside_mae=float(diff[:,~guard].mean()) if (~guard).any() else 0.
                outside_fraction=float(changed[~guard].mean()) if (~guard).any() else 0.
                ok=outside_mae<=cfg.max_outside_mae and outside_fraction<=cfg.max_outside_changed_fraction
                accepted&=ok
                item={'camera':cam.name,'visible_before':float(visible_before[mask].sum()/max(mass,1e-8)),
                    'visible_after':float(visible_after[mask].sum()/max(mass,1e-8)),
                    'outside_mask_mae':float(diff[:,~mask].mean()),'outside_guard_mae':outside_mae,
                    'outside_mask_changed_fraction':float(changed[~mask].mean()),'outside_guard_changed_fraction':outside_fraction,
                    'outside_guard_pass':ok,'remaining_occlusion_fraction':float(1-visible_after[mask].sum()/max(mass,1e-8))}
                metrics.append(item)
                if save:
                    c.save_image(arrays/f'{vi:02d}_before.png',before); c.save_image(arrays/f'{vi:02d}_after.png',after)
                    c.save_image(arrays/f'{vi:02d}_comparison.png',torch.cat((before,after),dim=2))
                    c.save_image(arrays/f'{vi:02d}_outside_change.png',diff.max(0)*(~mask))
        del filtered,joint; torch.cuda.empty_cache()
        return accepted,metrics
    proposed=len(selected); rollbacks=[]
    while True:
        passed,metrics=evaluate(selected)
        if passed: break
        rollbacks.append({'count':len(selected),'view_metrics':metrics})
        # Keep the highest benefit/collateral ranked half; never relax guards.
        selected=selected[:len(selected)//2]
    _,metrics=evaluate(selected,save=True)
    removed=ids[selected]; keep=np.ones(len(vertices),bool); keep[removed]=False
    save_vertices(output_dir/'background_filtered.ply',vertices[keep])
    save_vertices(output_dir/'removed_occluders.ply',vertices[removed])
    np.save(output_dir/'removed_background_indices.npy',removed)
    np.save(output_dir/'retained_background_indices.npy',np.flatnonzero(keep))
    significant=(gain/np.maximum(np.asarray(masses)[:,None],1)>=cfg.min_occlusion_fraction).any(0)
    unresolved_mask=significant&~np.isin(np.arange(len(ids)),selected)
    unresolved=ids[unresolved_mask]
    np.save(output_dir/'unresolved_candidate_indices.npy',unresolved)
    compose(output_dir/'background_filtered.ply',object_path,output_dir/'scene_filtered.ply',output_dir/'composition.json')
    complete=read_vertices(output_dir/'scene_filtered.ply')
    assert complete[:keep.sum()].tobytes()==vertices[keep].tobytes()
    assert complete[keep.sum():].tobytes()==object_vertices.tobytes()
    assert inputs['background']==sha256(background_path) and inputs['object']==sha256(object_path)
    report={'method':'OAGOF','version':2,'status':'validated','config':asdict(cfg),'input_sha256':inputs,
        'spatial_context_sha256':spatial_context_digest(edit_region,cameras),
        'input_paths':{'background':str(background_path),'object':str(object_path),'reference_mask':str(reference_mask)},
        'background_count':len(vertices),'object_count':len(object_vertices),'spatial_candidate_count':len(ids),
        'proposed_deletions':proposed,'removed_count':len(removed),'removed_fraction':len(removed)/len(vertices),
        'new_gaussians_count':0,'floor_repair':False,'object_bitwise_unchanged':True,'retained_background_bitwise_unchanged':True,
        'unresolved_candidate_count':len(unresolved),'unresolved_policy':'Retain insufficient multi-view/depth/support evidence or excessive outside-mask impact; remaining occlusion can be legitimate foreground.',
        'unresolved_reasons':{'insufficient_confirmed_views':int(np.sum(unresolved_mask&(votes.sum(0)<(cfg.min_confirm_views if cfg.use_multi_view else 1)))),
            'outside_impact_or_deletion_budget':int(np.sum(unresolved_mask&(votes.sum(0)>=(cfg.min_confirm_views if cfg.use_multi_view else 1))))},
        'view_setup':view_reports,'view_metrics':metrics,'guard_rollbacks':rollbacks,'seconds':time.monotonic()-start,
        'output_sha256':{name:sha256(output_dir/name) for name in ('background_filtered.ply','removed_occluders.ply','scene_filtered.ply')},
        'limitations':'This is a conservative geometric/visibility heuristic, not a semantic proof that every nearby occluder is erroneous. No ground completion or device-exterior black-smear cleanup.'}
    write_json(output_dir/'occlusion_report.json',report)
    print(json.dumps({k:report[k] for k in ('removed_count','unresolved_candidate_count','seconds','view_metrics')},indent=2),flush=True)
    return report
