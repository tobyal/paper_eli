"""Visible old-device silhouette plus depth-guarded replacement-space growth."""
import cv2
import numpy as np
from scipy import ndimage
import torch


def ray_box(camera, frame):
    yy,xx=np.mgrid[:camera.height,:camera.width]
    directions=np.stack(((xx-camera.cx)/camera.fx,(yy-camera.cy)/camera.fy,np.ones_like(xx)),axis=-1)@camera.R
    axes=np.asarray(frame['axes']); center=np.asarray(frame['center'])
    origin=(camera.center-center)@axes; directions=directions@axes
    lo=np.asarray(frame['local_bbox_low']); hi=np.asarray(frame['local_bbox_high'])
    near=np.full(xx.shape,-np.inf); far=np.full(xx.shape,np.inf); parallel_outside=np.zeros(xx.shape,bool)
    for k in range(3):
        parallel=np.abs(directions[...,k])<1e-9
        parallel_outside |= parallel & ((origin[k]<lo[k])|(origin[k]>hi[k]))
        a=np.divide(lo[k]-origin[k],directions[...,k],out=np.zeros(xx.shape),where=~parallel)
        b=np.divide(hi[k]-origin[k],directions[...,k],out=np.zeros(xx.shape),where=~parallel)
        near=np.maximum(near,np.where(parallel,-np.inf,np.minimum(a,b)))
        far=np.minimum(far,np.where(parallel,np.inf,np.maximum(a,b)))
    return (far>np.maximum(near,.01))&~parallel_outside,np.maximum(near,.01),far


def cleanup(mask, cfg):
    area=max(int(mask.sum()),1)
    labels,count=ndimage.label(mask)
    sizes=np.bincount(labels.ravel()); keep=sizes>=max(2,area*cfg['component_area_fraction']); keep[0]=False
    clean=keep[labels]
    radius=max(1,round(np.sqrt(area)*cfg['edge_fraction']))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    clean=cv2.morphologyEx(clean.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
    holes=ndimage.binary_fill_holes(clean)&~clean
    labels,count=ndimage.label(holes); sizes=np.bincount(labels.ravel())
    fill=sizes<=area*cfg['hole_area_fraction']; fill[0]=False
    clean |= fill[labels]
    return clean,radius


def swept_growth(camera, clean, region, radius):
    """Sweep the measured silhouette only by positive changes in 3D extents.

    This is a shape envelope, not a generated device or a filled box projection.
    """
    geometry=region['geometry']; axes=np.asarray(geometry['axes'])
    origin=np.asarray(geometry['origin']); old_low=np.asarray(geometry['old_local_low'])
    old_high=np.asarray(geometry['old_local_high']); new_low=np.asarray(geometry['new_local_low'])
    new_high=np.asarray(geometry['new_local_high']); center=origin+axes@((old_low+old_high)/2)
    start,_=camera.project(center[None]); vectors=[]
    for axis in range(3):
        for sign,amount in ((-1,max(0,old_low[axis]-new_low[axis])),(1,max(0,new_high[axis]-old_high[axis]))):
            if amount>0:
                end,z=camera.project((center+sign*amount*axes[:,axis])[None])
                if z[0]>.01: vectors.append((end-start)[0])
    grown=clean.astype(np.uint8)
    for vector in vectors:
        source=grown.copy()
        steps=max(1,int(np.ceil(np.linalg.norm(vector)/2)))
        for t in np.linspace(0,1,min(steps,512)+1)[1:]:
            transform=np.array([[1,0,t*vector[0]],[0,1,t*vector[1]]],np.float32)
            grown |= cv2.warpAffine(source,transform,(camera.width,camera.height),flags=cv2.INTER_NEAREST)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    grown=cv2.dilate(grown,kernel).astype(bool)
    return grown,[v.tolist() for v in vectors]


def build_mask(camera, maps, region, box_mask, cfg):
    size=region['object_scale']; margin=size*cfg['depth_margin_fraction']
    independent=maps['target_alpha']; visible=maps['target_visible_alpha']
    ratio=visible/np.maximum(independent,1e-6)
    # Joint contribution is exact Gaussian compositing; depth is a second consistency check.
    source_depth_ok=(maps['target_depth'] <= maps['scene_depth'] +
        2*(maps['target_std']+maps['scene_std'])+margin) | (visible>.5)
    raw=(independent>cfg['old_alpha_threshold'])&(visible>cfg['old_alpha_threshold'])&(
        ratio>cfg['minimum_visible_fraction'])&source_depth_ok
    if raw.sum()<16:
        raise ValueError('Anchor has insufficient visible old-target support; inspect another AVP view')
    clean,radius=cleanup(raw,cfg)
    hit,near,far=ray_box(camera,region['world_transform'])
    bg_z=maps['background_depth']; bg_std=maps['background_std']
    reliable=(maps['background_alpha']>cfg['background_alpha_threshold'])&(bg_std<size*cfg['depth_std_fraction'])
    in_front=hit&reliable&(bg_z+2*bg_std<near-margin)
    if 'background_front_mass' in maps:
        # A broad depth distribution behind the editing volume is not an occluder.
        # Query the rendered depth CDF at the volume's front and back surfaces.
        in_front=hit&(maps['background_front_mass']>cfg['protected_contribution'])
        reliable=(maps['background_alpha']>cfg['background_alpha_threshold'])&(
            maps['background_front_mass']<cfg['max_front_contribution'])
        bg_z=maps['background_surface_depth']
    # Also preserve a foreground rail passing through the old silhouette.
    old_occluder=(independent>cfg['old_alpha_threshold'])&(maps['background_alpha']>cfg['background_alpha_threshold'])&(
        ratio<cfg['minimum_visible_fraction'])&(bg_z+2*bg_std<maps['target_depth']-margin)
    # Dense equipment physically inside the proposed box is a collision, not free space.
    yy,xx=np.mgrid[:camera.height,:camera.width]
    rays=np.stack(((xx-camera.cx)/camera.fx,(yy-camera.cy)/camera.fy,np.ones_like(xx)),axis=-1)@camera.R
    world=camera.center+rays*bg_z[...,None]
    frame=region['world_transform']; up=np.asarray(region['up'])
    height=(world-np.asarray(frame['center']))@up-float(frame['local_bbox_low'][1])
    independent_guard=cv2.dilate((independent>cfg['old_alpha_threshold']).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
    occupied=hit&reliable&~independent_guard&(bg_z-2*bg_std>near-margin)&(bg_z+2*bg_std<far-margin)&(
        height>region['geometry']['new_side_lengths'][1]*.1)
    if 'background_volume_mass' in maps:
        occupied=hit&~independent_guard&(maps['background_volume_mass']>cfg['protected_contribution'])&(
            height>region['geometry']['new_side_lengths'][1]*.1)
        old_occluder=(independent>cfg['old_alpha_threshold'])&(
            ratio<cfg['minimum_visible_fraction'])&(maps['background_old_front_mass']>cfg['protected_contribution'])
    protected=in_front|old_occluder|occupied
    clean &= ~protected
    proposal,vectors=swept_growth(camera,clean,region,radius)
    # Low confidence depth never authorizes new pixels. The measured visible seed remains.
    extension=proposal&box_mask&hit&reliable&~protected&~clean
    final=clean|extension
    labels,count=ndimage.label(final)
    connected=np.unique(labels[clean]); keep=np.zeros(count+1,bool); keep[connected]=True; keep[0]=False
    final=keep[labels]
    reference=visible>cfg['old_alpha_threshold']
    stats={'old_visible_pixels':int(reference.sum()),'old_depth_checked_pixels':int(raw.sum()),
        'old_clean_pixels':int(clean.sum()),'edit_pixels':int(final.sum()),
        'generation_projection_pixels':int(box_mask.sum()),'extension_pixels':int((final&~clean).sum()),
        'protected_foreground_pixels':int((protected&proposal).sum()),
        'protected_mask_overlap_pixels':int((protected&final).sum()),
        'uncertain_extension_rejected_pixels':int((proposal&box_mask&~reliable&~clean).sum()),
        'visible_alpha_mass_retained':float(visible[final].sum()/max(float(visible.sum()),1e-8)),
        'visible_mask_coverage':float((final&reference).sum()/max(int(reference.sum()),1)),
        'edge_radius_pixels':radius,'sweep_vectors_pixels':vectors,
        'outside_generation_pixels':int((final&~box_mask).sum()),
        'outside_generation_policy':'Only cleaned old-target removal silhouette may extend beyond new-object space'}
    desired_extension=int((proposal&box_mask&~clean).sum())
    stats['desired_extension_pixels']=desired_extension
    stats['extension_fraction_available']=float(stats['extension_pixels']/max(desired_extension,1))
    warnings=[]
    if stats['visible_alpha_mass_retained']<.95:warnings.append('Insufficient visible old-target contribution retained')
    if desired_extension>clean.sum()*.1 and stats['extension_fraction_available']<.15:
        warnings.append('Required replacement growth mostly blocked; review device size or Anchor View')
    stats['quality_warnings']=warnings
    stats['quality_status']='manual_review_required' if warnings else 'passed_geometry_checks'
    masks={'old_visible_mask':reference,'old_clean_mask':clean,'extension_proposal':proposal,
           'protected_foreground_mask':protected,'generation_bbox_mask':box_mask,'edit_mask':final}
    return final,masks,stats


def render_geometry(c,camera,scene,ids,region=None,cfg=None):
    """Camera-z first/second moments, distinct from the legacy display depth."""
    index=torch.as_tensor(ids,device=scene.xyz.device)
    keep=torch.ones(len(scene.xyz),dtype=torch.bool,device=scene.xyz.device); keep[index]=False
    target=scene.subset(index); background=scene.subset(keep)
    maps={}
    with torch.no_grad():
        for name,data in (('scene',scene),('target',target),('background',background)):
            z=data.xyz@torch.as_tensor(camera.R[2],device=data.xyz.device)+float(camera.t[2])
            packed=c.raster(camera,data,torch.stack((torch.ones_like(z),z,z*z),dim=1))
            alpha=packed[0]; depth=packed[1]/alpha.clamp_min(1e-6)
            std=(packed[2]/alpha.clamp_min(1e-6)-depth*depth).clamp_min(0).sqrt()
            for suffix,value in (('alpha',alpha),('depth',depth),('std',std)):
                maps[name+'_'+suffix]=value.cpu().numpy()
        flags=torch.zeros_like(scene.xyz); flags[index]=1
        maps['target_visible_alpha']=c.raster(camera,scene,flags)[0].cpu().numpy()
        if region is not None:
            hit,near,far=ray_box(camera,region['world_transform'])
            margin=region['object_scale']*cfg['depth_margin_fraction']
            _,corner_z=camera.project(np.asarray(region['bbox_corners']))
            edges=np.linspace(max(.001,float(corner_z.min())-3*margin),float(corner_z.max())+3*margin,cfg['depth_cdf_bins'])
            z=background.xyz@torch.as_tensor(camera.R[2],device=scene.xyz.device)+float(camera.t[2])
            cdf=[]
            for start in range(0,len(edges),3):
                group=edges[start:start+3]; thresholds=np.pad(group,(0,3-len(group)),mode='edge')
                colors=(z[:,None]<=torch.as_tensor(thresholds,device=z.device)[None]).float()
                packed=c.raster(camera,background,colors).cpu().numpy()
                cdf.extend(packed[:len(group)])
            cdf=np.asarray(cdf)
            def query(depth):
                indices=np.clip(np.searchsorted(edges,depth,side='right'),1,len(edges)-1)
                a=np.take_along_axis(cdf,(indices-1)[None],axis=0)[0]
                b=np.take_along_axis(cdf,indices[None],axis=0)[0]
                fraction=np.clip((depth-edges[indices-1])/(edges[indices]-edges[indices-1]),0,1)
                return (a*(1-fraction)+b*fraction).astype(np.float32)
            # Before-near and before-far are conservative upper/lower CDF queries;
            # rendering a white depth-prefix measures its actual transmittance contribution.
            maps['background_front_mass']=query(near-margin)
            maps['background_volume_mass']=np.maximum(query(far-margin)-query(near+margin),0)
            maps['background_old_front_mass']=query(maps['target_depth']-margin)
            median=(cdf>=maps['background_alpha'][None]*.5).argmax(0)
            maps['background_surface_depth']=edges[median].astype(np.float32)
            maps['background_surface_depth'][cdf[-1]<maps['background_alpha']*.5]=edges[-1]
    return maps
