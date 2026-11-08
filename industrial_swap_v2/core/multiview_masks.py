"""M4 dual-mask supervision and occlusion attribution. Never alters/deletes splats."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from .context import gsle, read_json, write_json, sha256
from .scene import read_vertices
from .visibility_loss import object_contribution
from .workflow import supervision_settings
from .scene_integration import object_path


DEFAULTS={'enabled':True,'resolution':640,'source_views':40,'holdout_every':4,
          'local_expansion':.25,'min_observed_views':2,
          'min_observed_pixels':.5,'candidate_contribution_fraction':1e-5,
          'depth_margin_fraction':.015,'depth_uncertainty_fraction':.08,
          'scheme':'old_new_union_v2','old_alpha_threshold':.02,'new_alpha_threshold':.25,
          'union_edge_fraction':.003,'occlusion_expansion':.08,
          'min_occlusion_views':3,'min_occlusion_angle_degrees':5.,
          'min_gain_to_environment':.03,'max_legal_fraction':.02}


UNION_SCHEME='old_new_union_v2'


def union_supervision_masks(camera,box,source_depth,source_std,source_alpha,old_alpha,
                           obj_depth,obj_alpha,size,cfg):
    """BBox is used only for conservative foreground protection, never RGB exclusion."""
    old=old_alpha>cfg['old_alpha_threshold']; new=obj_alpha>cfg['new_alpha_threshold']
    excluded=old|new; env=~excluded
    radius=max(0,round(min(camera.width,camera.height)*cfg['union_edge_fraction']))
    guarded=cv2.dilate(excluded.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    photo=env&~guarded
    photo[:2]=False; photo[-2:]=False; photo[:,:2]=False; photo[:,-2:]=False
    hit,near,_=ray_box_depth(camera,*box); margin=size*cfg['depth_margin_fraction']
    confident=(source_alpha>.95)&(source_std<size*cfg['depth_uncertainty_fraction'])
    front_of_old=(~hit)|(source_depth+2*source_std<near-margin)
    legal=new&confident&(old_alpha<.01)&front_of_old&(source_depth+2*source_std<obj_depth-margin)
    visibility=new&~legal
    return old,new,env,photo,legal,visibility


def occlusion_permissions(spatial_ids,votes,directions,legal_veto,cfg):
    """No RGB-observation condition: multi-view negative-opacity evidence only."""
    votes=np.asarray(votes,dtype=bool); directions=np.asarray(directions)
    if len(votes)==0: return np.empty(0,np.int64),np.zeros(len(spatial_ids))
    span=np.zeros(len(spatial_ids),np.float32)
    dots=np.clip(directions@directions.T,-1,1)
    for i in range(len(votes)):
        for j in range(i):
            angle=float(np.degrees(np.arccos(dots[i,j])))
            span=np.maximum(span,np.where(votes[i]&votes[j],angle,0))
    selected=(votes.sum(0)>=cfg['min_occlusion_views'])&(span>=cfg['min_occlusion_angle_degrees'])&~legal_veto
    return np.asarray(spatial_ids)[selected],span


def ray_box_depth(camera,low,high):
    """World AABB ray interval, parameterized by camera z, not Euclidean depth."""
    yy,xx=np.indices((camera.height,camera.width)); rays=np.stack(((xx-camera.cx)/camera.fx,
        (yy-camera.cy)/camera.fy,np.ones_like(xx)),axis=-1)@camera.R
    safe=np.where(abs(rays)<1e-10,np.where(rays>=0,1e-10,-1e-10),rays)
    a=(np.asarray(low)-camera.center)/safe; b=(np.asarray(high)-camera.center)/safe
    near=np.maximum(np.minimum(a,b).max(-1),0); far=np.maximum(a,b).min(-1)
    hit=(far>near)&(far>0)
    return hit,near,far




def local_support(vertices,manifest,expansion):
    xyz=np.column_stack([vertices[n] for n in ('x','y','z')]); frame=manifest['world_transform']
    axes=np.asarray(frame['axes']); center=np.asarray(frame['center'])
    local=(xyz-center)@axes; low=np.asarray(frame['local_bbox_low']); high=np.asarray(frame['local_bbox_high'])
    size=float(np.linalg.norm(high-low)); scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]))
    q=np.column_stack([vertices[f'rot_{i}'] for i in (1,2,3,0)])
    rot=Rotation.from_quat(q).as_matrix()
    projected=np.einsum('ij,njk->nik',axes.T,rot)
    std=np.sqrt(np.einsum('nik,nk->ni',projected**2,scales**2))
    gap=np.maximum(np.maximum(low-expansion*size-local,local-high-expansion*size),0)
    support=(gap<3*std).all(1)&(np.linalg.norm(local,axis=1)<2*size)&(scales.max(1)<size)
    return np.flatnonzero(support),size


def choose_views(cameras,manifest,count):
    corners=np.asarray(manifest['target_bbox']); center=corners.mean(0); candidates=[]
    for camera in cameras:
        xy,z=camera.project(np.r_[corners,center[None]])
        x,y=xy[-1]
        if (z>0).all() and 0<x<camera.width and 0<y<camera.height:
            area=np.ptp(xy[:8],axis=0).prod()/(camera.width*camera.height)
            if .002<area<.65: candidates.append(camera)
    if len(candidates)<8: raise ValueError('Insufficient registered photo views of the old region')
    return [candidates[i] for i in np.unique(np.linspace(0,len(candidates)-1,min(count,len(candidates))).astype(int))]


def prepare_multiview(ctx,output_dir=None):
    cfg=mask_config(supervision_settings(ctx.cfg))
    if cfg['source_views']<8 or cfg['holdout_every']<2 or cfg['min_observed_views']<2:
        raise ValueError('Multiview optimization needs independent train/validation views and at least two observation votes')
    if not 0<=cfg['local_expansion']<=1 or not 0<=cfg['union_edge_fraction']<.1:
        raise ValueError('Invalid relative spatial/mask margin')
    out=Path(output_dir or ctx.stage('m4')/'multiview_supervision')
    sources={'background':ctx.stage('m1')/'background_gaussians.ply','object':object_path(ctx),
        'manifest':ctx.stage('m1')/'manifest.json','mask':ctx.stage('m2')/'object_mask.png',
        'camera':ctx.stage('m2')/'anchor_camera.json','original_scene':ctx.source(ctx.cfg['project']['source_ply']),
        'target_indices':ctx.stage('m1')/'target_indices.npy'}
    signature=hashlib.sha256(json.dumps({'config':cfg,'hashes':{k:sha256(v) for k,v in sources.items()}},sort_keys=True).encode()).hexdigest()
    if sha256(sources['original_scene'])!=ctx.cfg['project']['source_sha256']:
        raise ValueError('Source reconstruction differs from the M1 project fingerprint')
    if (out/'manifest.json').exists():
        saved=read_json(out/'manifest.json')
        if not cache_matches(saved,cfg,{k:sha256(v) for k,v in sources.items()}):
            raise ValueError('M4 supervision input/config mismatch; use a new output directory')
        return out
    out.mkdir(parents=True,exist_ok=True)
    for name in ('cache','object_masks','object_alpha','object_depth','renders','supervision_masks'): (out/name).mkdir(exist_ok=True)
    c=gsle(ctx); manifest=read_json(sources['manifest']); vertices=read_vertices(sources['background'])
    local_ids,size=local_support(vertices,manifest,cfg['local_expansion'])
    spatial_ids,_=local_support(vertices,manifest,cfg['occlusion_expansion'])
    vote_matrix=[]; vote_directions=[]; legal_veto=np.zeros(len(spatial_ids),bool)
    bg=c.GaussianData.load(sources['background']); obj=c.GaussianData.load(sources['object'])
    source=c.GaussianData.load(sources['original_scene']); old_indices=np.load(sources['target_indices'])
    flags=torch.zeros((len(source.xyz),3),device='cuda'); flags[torch.tensor(old_indices,device='cuda')]=1
    anchor=c.Camera.from_record(read_json(sources['camera'])); width=cfg['resolution']
    anchor=anchor.resize(width,round(anchor.height*width/anchor.width))
    cameras=[anchor,*choose_views(c.source_cameras(width),manifest,cfg['source_views'])]
    counts=np.zeros(len(vertices),np.int16); observed_mass=np.zeros(len(vertices),np.float32)
    votes=np.zeros(len(vertices),np.int16); score=np.zeros(len(vertices),np.float32); records=[]
    old_corners=np.asarray(manifest['target_bbox']); box=(old_corners.min(0),old_corners.max(0))
    local_flags=torch.zeros((len(vertices),3),device='cuda'); local_flags[torch.tensor(local_ids,device='cuda')]=1
    for index,camera in enumerate(cameras):
        name=f'{index:03d}_{Path(camera.name).stem}'; split='anchor' if index==0 else 'validation' if (index-1)%cfg['holdout_every']==0 else 'train'
        print('multiview',index+1,len(cameras),camera.name,split,flush=True)
        with torch.no_grad():
            z=source.xyz@torch.tensor(camera.R[2],device='cuda')+float(camera.t[2])
            moments=c.raster(camera,source,torch.stack((torch.ones_like(z),z,z*z),dim=1))
            sa=moments[0]; sd=moments[1]/sa.clamp_min(1e-6)
            ss=(moments[2]/sa.clamp_min(1e-6)-sd*sd).clamp_min(0).sqrt()
            old=c.raster(camera,source,flags)[0]
            zo=obj.xyz@torch.tensor(camera.R[2],device='cuda')+float(camera.t[2])
            op=c.raster(camera,obj,torch.stack((torch.ones_like(zo),zo,torch.zeros_like(zo)),dim=1))
            oa=op[0]; od=op[1]/oa.clamp_min(1e-6)
            object_rgb=c.render(camera,obj,aux=False)['rgb']; background_rgb=c.render(camera,bg,aux=False)['rgb']
            joint_rgb=c.render_joint(camera,bg,obj,aux=False)['rgb']
            joint=c.concatenate(bg,obj); zj=joint.xyz@torch.tensor(camera.R[2],device='cuda')+float(camera.t[2])
            joint_flags=torch.cat((torch.zeros(len(bg.xyz),device='cuda'),torch.ones(len(obj.xyz),device='cuda')))
            jp=c.raster(camera,joint,torch.stack((torch.ones_like(zj),zj,joint_flags),dim=1))
            joint_depth=jp[1]/jp[0].clamp_min(1e-6); visible=jp[2]
            local_alpha=c.raster(camera,bg,local_flags)[0].cpu().numpy()
        cpu=lambda tensor:tensor.detach().cpu().numpy()
        mold,mnew,menv,reliable,legal,wvis=union_supervision_masks(camera,box,cpu(sd),cpu(ss),cpu(sa),cpu(old),cpu(od),cpu(oa),size,cfg)
        unknown=mold|mnew; old_projection=mold
        reference_iou=None
        if split=='anchor':
            with Image.open(sources['mask']) as image: reference=np.asarray(image.convert('L').resize((camera.width,camera.height),Image.Resampling.NEAREST))>127
            modelmask=cpu(oa)>.25; reference_iou=float((reference&modelmask).sum()/max((reference|modelmask).sum(),1))
            c.save_image(out/'object_masks'/f'{name}_m2_reference.png',reference.astype(float))
            original=np.zeros((camera.height,camera.width,3),np.uint8); reliable[:]=False
        else:
            image_path=ctx.source('data/industrial/images')/camera.name
            with Image.open(image_path) as image: original=np.asarray(image.convert('RGB').resize((camera.width,camera.height),Image.Resampling.LANCZOS))
        train_mask=reliable&(local_alpha>.01)
        # RGB observation permission and occlusion permission are independent.
        if split in ('train','anchor'):
            colors=torch.zeros((len(bg.xyz),3),device='cuda',requires_grad=True)
            rendered=c.raster(camera,bg,colors)
            (rendered*torch.tensor(np.stack((menv,legal,wvis)),device='cuda')).sum().backward()
            masses=colors.grad.detach().cpu().numpy()
            mass=masses[:,0]
            threshold=cfg['min_observed_pixels']*camera.width*camera.height/(640*360)
            if split=='train':
                # The local 3D set limits membership; zero weights stay excluded.
                colors.grad.zero_(); rendered=c.raster(camera,bg,colors)
                (rendered[0]*torch.tensor(train_mask,device='cuda')).sum().backward()
                mass=colors.grad[:,0].detach().cpu().numpy().copy()
                counts+=(mass>threshold); observed_mass+=mass
            del colors,rendered
        if split in ('train','anchor'):
            bg.opacity_logit=bg.opacity_logit.detach().requires_grad_(True)
            contribution=object_contribution(c,camera,bg,obj)[0]
            (contribution*torch.tensor(wvis,device='cuda')).sum().backward()
            alpha=bg.opacity_logit.detach().sigmoid()[:,0].cpu().numpy()
            gain=np.maximum(-bg.opacity_logit.grad[:,0].detach().cpu().numpy(),0)/np.maximum(1-alpha,1e-6)
            relative=gain/max(float((cpu(oa)*wvis).sum()),1)
            votes+=(relative>cfg['candidate_contribution_fraction']); score+=relative
            outside=masses[spatial_ids,0]; inside=masses[spatial_ids,2]; protected=masses[spatial_ids,1]
            # Actual visibility derivative proves front-to-back occlusion;
            # 3-sigma support and spatial locality are checked by spatial_ids.
            evidence=(relative[spatial_ids]>cfg['candidate_contribution_fraction'])&(gain[spatial_ids]/np.maximum(outside,1e-6)>=cfg['min_gain_to_environment'])
            legal_veto|=(protected/np.maximum(inside+protected,1e-6)>cfg['max_legal_fraction'])&(protected>threshold)
            vote_matrix.append(evidence)
            direction=camera.center-np.asarray(manifest['world_transform']['center'])
            vote_directions.append(direction/np.linalg.norm(direction))
            bg.opacity_logit=bg.opacity_logit.detach(); del contribution
        regions={'overlap':mold&mnew,'old_only':mold&~mnew,'new_only':mnew&~mold,
                 'new_visible':cpu(visible)>cfg['new_alpha_threshold'],
                 'new_occluded':mnew&(cpu(visible)<cpu(oa)*.9)}
        extra={'old_mask':mold,'new_mask':mnew,'environment_mask':menv,**regions}
        np.savez_compressed(out/'cache'/f'{name}.npz',**extra,original_rgb=original,background_initial=cpu(background_rgb),
            object_alpha=cpu(oa),object_depth=cpu(od),joint_depth=cpu(joint_depth),visible_initial=cpu(visible),
            reliable_background=reliable,background_train_mask=train_mask,visibility_weight=wvis,
            unknown_region=unknown,legal_foreground=legal,old_projection=old_projection,old_alpha=cpu(old),local_alpha=local_alpha)
        c.save_image(out/'object_masks'/f'{name}.png',(cpu(oa)>.25).astype(float)); c.save_image(out/'object_alpha'/f'{name}.png',oa)
        np.save(out/'object_depth'/f'{name}.npy',cpu(od))
        for label,tensor in [('object',object_rgb),('background',background_rgb),('joint',joint_rgb),('visible',visible)]:
            c.save_image(out/'renders'/f'{name}_{label}.png',tensor)
        for label,mask in [('reliable',reliable),('unknown',unknown),('visibility',wvis),('legal_foreground',legal)]:
            c.save_image(out/'supervision_masks'/f'{name}_{label}.png',mask.astype(float))
        for label,mask in [('old',mold),('new',mnew),('environment',menv),('environment_loss',train_mask)]:
            c.save_image(out/'supervision_masks'/f'{name}_{label}.png',mask.astype(float))
        for label,mask in regions.items():
            c.save_image(out/'supervision_masks'/f'{name}_{label}.png',mask.astype(float))
        records.append({'name':name,'camera':camera.record(),'split':split,'original_image':None if split=='anchor' else str(image_path),
            'reference_iou':reference_iou,'reliable_pixels':int(reliable.sum()),'local_reliable_pixels':int(train_mask.sum()),
            'unknown_pixels':int(unknown.sum()),'visibility_pixels':int(wvis.sum()),'legal_foreground_pixels':int(legal.sum()),
            'region_pixel_counts':{k:int(mask.sum()) for k,mask in regions.items()}})
        del moments,op,jp,joint,z,zj,zo,sa,sd,ss,old,oa,od,object_rgb,background_rgb,joint_rgb,visible,local_alpha
        torch.cuda.empty_cache()
    supported=counts[local_ids]>=cfg['min_observed_views']; optimizable=local_ids[supported]
    rgb_indices=optimizable.copy()
    candidates,span=occlusion_permissions(spatial_ids,vote_matrix,vote_directions,legal_veto,cfg)
    optimizable=np.union1d(rgb_indices,candidates); unsupported=np.setdiff1d(local_ids,optimizable)
    np.save(out/'rgb_indices.npy',rgb_indices); np.save(out/'occlusion_indices.npy',candidates)
    np.savez_compressed(out/'occlusion_evidence.npz',spatial_indices=spatial_ids,votes=np.asarray(vote_matrix),
        angular_span_degrees=span,legal_foreground_veto=legal_veto)
    np.save(out/'local_indices.npy',optimizable); np.save(out/'unsupported_local_indices.npy',unsupported)
    np.save(out/'occlusion_candidate_indices.npy',candidates)
    np.savez_compressed(out/'observational_support.npz',counts=counts,mass=observed_mass,occlusion_votes=votes,occlusion_score=score)
    write_json(out/'occlusion_candidates.json',{'analysis_only':True,'deleted_gaussians':0,'candidate_count':len(candidates),
        'supported_candidate_count':int(np.isin(candidates,rgb_indices).sum()),'local_supported_count':len(optimizable),
        'local_unsupported_frozen_count':len(unsupported),'top_candidates':[{'background_index':int(i),'score':float(score[i]),'views':int(votes[i]),'observed_views':int(counts[i])} for i in candidates[np.argsort(-score[candidates])[:50]]]})
    write_json(out/'manifest.json',{'signature':signature,'config':cfg,'sources':{k:str(v) for k,v in sources.items()},
        'source_hashes':{k:sha256(v) for k,v in sources.items()},'views':records,'region_scale':size,
        'permission_scheme':cfg['scheme'],
        'rgb_count':len(rgb_indices),
        'occlusion_count':len(candidates),
        'occlusion_without_rgb_count':int((~np.isin(candidates,rgb_indices)).sum()),
        'local_supported_count':len(optimizable),'local_unsupported_count':len(unsupported),
        'semantics':('M_env = complement of visible old semantic mask union independently rendered new mask. RGB and occlusion permissions are separate; depth is used only for foreground protection.'),
        'unknown_policy':('RGB excludes the union of old visible semantic mask and new isolated alpha mask; occlusion-only splats may change opacity without RGB support, subject to spatial/multiview/foreground guards; no completion.')})
    return out


def mask_config(values=None):
    # Compatibility for the earlier union cache: this bbox margin was unused.
    values=dict(values or {}); values.pop('edge_dilation_fraction',None)
    cfg={**DEFAULTS,**values}
    if cfg['scheme']!=UNION_SCHEME or not cfg['enabled']:
        raise ValueError('Official M4 requires old_new_union_v2 masks')
    if set(cfg)-set(DEFAULTS): raise ValueError('Unknown multiview configuration keys')
    return cfg


def cache_matches(saved,cfg,source_hashes):
    try:
        return (saved.get('permission_scheme')==UNION_SCHEME and
                mask_config(saved['config'])==cfg and saved['source_hashes']==source_hashes)
    except (ValueError,KeyError):
        return False


def ensure_m4_supervision(ctx):
    """Build new supervision only in M4; reuse matching historical caches read-only."""
    cfg=mask_config(supervision_settings(ctx.cfg))
    sources={'background':ctx.stage('m1')/'background_gaussians.ply',
        'object':object_path(ctx),'manifest':ctx.stage('m1')/'manifest.json',
        'mask':ctx.stage('m2')/'object_mask.png','camera':ctx.stage('m2')/'anchor_camera.json',
        'original_scene':ctx.source(ctx.cfg['project']['source_ply']),
        'target_indices':ctx.stage('m1')/'target_indices.npy'}
    hashes={k:sha256(v) for k,v in sources.items()}
    destination=ctx.stage('m4')/'multiview_supervision'
    candidates=(destination,ctx.stage('m4')/'background_optimization/visibility_prepare',
                ctx.stage('m3')/'multiview_visibility')
    for folder in candidates:
        if not (folder/'manifest.json').exists(): continue
        if cache_matches(read_json(folder/'manifest.json'),cfg,hashes): return folder
        if folder==destination:
            raise ValueError('M4 supervision input/config mismatch; use a new output directory')
    return prepare_multiview(ctx,destination)
