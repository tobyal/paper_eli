"""Four-parameter, differentiable Gaussian registration to a saved M2 camera.

Every image loss uses all Gaussians and their actual covariance/opacity. Only
uniform scale, yaw, and two horizontal translations are optimized. SH remains
in the local frame during fitting by evaluating directions in that frame;
export rotates all SH bands, covariance orientations, scales and positions.
"""
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from .context import gsle,read_json,write_json,sha256
from .scene import read_vertices,save_vertices
from .alignment import transform_vertices
from .m2_space import registration_space,weighted_quantile

DEFAULTS=dict(search_resolution=320,refine_resolution=640,yaw_step_degrees=15,
    initial_size_fractions=[.75,.95,1.1],refine_candidates=6,coarse_steps=120,
    fine_steps=180,learning_rate=.012,max_horizontal_fraction=.3,
    min_scale_factor=.45,max_scale_factor=1.5,geometry_quantile=.002,
    silhouette_weight=1.,rgb_weight=.18,boundary_weight=.25,
    center_weight=.35,volume_weight=8.,max_volume_fraction=.12,
    min_silhouette_iou=.65)


def yaw_matrix(yaw):
    z=yaw*0;o=z+1;c=torch.cos(yaw);s=torch.sin(yaw)
    return torch.stack((c,z,s,z,o,z,-s,z,c)).reshape(3,3)


def quaternion_product(a,b):
    aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
    av,bv=torch.broadcast_tensors(av,bv)
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),dim=-1)


class Pose:
    def __init__(self,common,local,space,cfg):
        self.c,self.local,self.cfg=common,local,cfg
        self.tensor=lambda x:torch.as_tensor(x,device=local.xyz.device,dtype=torch.float32)
        frame=space['frame'];self.axes=self.tensor(frame['axes']);self.center=self.tensor(frame['center'])
        self.low=self.tensor(frame['local_bbox_low']);self.high=self.tensor(frame['local_bbox_high']);self.extent=self.high-self.low
        self.support=self.tensor(space['support_center_local'])
        self.axis_quat=self.tensor(Rotation.from_matrix(frame['axes']).as_quat()[[3,0,1,2]])
        # Robust geometry is only used for spatial constraints, never for masks.
        points=local.xyz.cpu().numpy();weights=local.opacity_logit.sigmoid().flatten().cpu().numpy()
        quats=F.normalize(local.rotation,dim=-1).cpu().numpy()
        matrices=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        scales=local.log_scale.exp().cpu().numpy()
        sigma=np.sqrt(np.sum(matrices**2*scales[:,None,:]**2,axis=2))
        cap=np.linalg.norm(np.ptp(points,axis=0))*.02;sigma=np.minimum(sigma,cap)
        q=cfg['geometry_quantile']
        lo=np.array([weighted_quantile(points[:,j]-sigma[:,j],weights,q) for j in range(3)])
        hi=np.array([weighted_quantile(points[:,j]+sigma[:,j],weights,1-q) for j in range(3)])
        pivot=(lo+hi)/2;pivot[1]=lo[1]
        self.pivot=self.tensor(pivot);self.bounds=(lo,hi)
        self.base_scale=float(np.min(self.extent.cpu().numpy()/np.maximum(hi-lo,1e-6)))
        idx=np.arange(0,len(points),max(1,len(points)//12000))
        self.probe=self.tensor(points[idx]);self.probe_weight=self.tensor(weights[idx])
        covariance=np.einsum('nij,nkj->nik',matrices[idx]*scales[idx,None,:],matrices[idx]*scales[idx,None,:])
        self.covariance=self.tensor(covariance);self.support_cap=cap

    def placement(self,p):
        ry=yaw_matrix(p[0]);scale=p[1].exp()
        offset=self.support+torch.stack((p[2]*self.extent[0],p[2]*0,p[3]*self.extent[2]))
        rotation=self.axes@ry;translation=self.center+self.axes@(offset-scale*(ry@self.pivot))
        return scale,rotation,translation,ry,offset

    def data(self,p):
        scale,rotation,translation,_,_=self.placement(p)
        half=p[0]/2;zero=half*0
        yaw_q=torch.stack((half.cos(),zero,half.sin(),zero))
        q=quaternion_product(self.axis_quat,yaw_q)
        quat=quaternion_product(q,F.normalize(self.local.rotation,dim=-1))
        return self.c.GaussianData(self.local.xyz@rotation.T*scale+translation,
            self.local.log_scale+p[1],quat,self.local.opacity_logit,self.local.sh)

    def render(self,p,camera):
        obj=self.data(p);scale,rotation,translation,_,_=self.placement(p)
        local_camera=(self.tensor(camera.center)-translation)@rotation/scale
        rgb=self.c.raster(camera,obj,self.local.colors(local_camera),torch.ones(3,device=p.device))
        z=obj.xyz@self.tensor(camera.R[2])+self.tensor(camera.t[2])
        packed=self.c.raster(camera,obj,torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),dim=1))
        alpha=packed[1].clamp(0,1)
        return {'rgb':rgb,'alpha':alpha,'depth_z':packed[0]/alpha.clamp_min(1e-6)}

    def spatial(self,p):
        scale,_,_,ry,offset=self.placement(p)
        pos=(self.probe-self.pivot)@ry.T*scale+offset
        variance=torch.einsum('ij,njk,ik->ni',ry,self.covariance,ry)
        sigma=variance.clamp_min(1e-12).sqrt().clamp_max(self.support_cap)*scale
        excess=(F.relu(self.low-(pos-sigma))+F.relu(pos+sigma-self.high))/self.extent
        volume=(excess.square().sum(1)*self.probe_weight).sum()/self.probe_weight.sum()
        return (p[2:].square().mean(),volume,pos,sigma)

    @torch.no_grad()
    def constrain(self,p):
        p[1].clamp_(math.log(self.base_scale*self.cfg['min_scale_factor']),math.log(self.base_scale*self.cfg['max_scale_factor']))
        p[2:].clamp_(-self.cfg['max_horizontal_fraction'],self.cfg['max_horizontal_fraction'])

    @torch.no_grad()
    def project_volume(self,p):
        """Project a candidate into the existing space gate, without relaxing it.

        Geometry/opacity stay frozen. This is a bounded pose update, used only
        when unconstrained reference fitting would fail the volume check.
        """
        self.constrain(p)
        scale,_,_,_,offset=self.placement(p);_,_,pos,sigma=self.spatial(p)
        lo=(torch.quantile(pos-sigma,.002,dim=0)-offset)/scale
        hi=(torch.quantile(pos+sigma,.998,dim=0)-offset)/scale
        tolerance=self.cfg['max_volume_fraction']*.995
        width_limit=(self.extent*(1+tolerance)/(hi-lo).clamp_min(1e-8)).min()
        height_limit=(self.high[1]+tolerance*self.extent[1]-self.support[1])/hi[1].clamp_min(1e-8)
        new_scale=torch.minimum(scale,torch.minimum(width_limit,height_limit))
        p[1].copy_(new_scale.log())
        for parameter,axis in ((2,0),(3,2)):
            minimum=self.low[axis]-tolerance*self.extent[axis]-new_scale*lo[axis]
            maximum=self.high[axis]+tolerance*self.extent[axis]-new_scale*hi[axis]
            target=offset[axis].clamp(minimum,maximum)
            p[parameter].copy_((target-self.support[axis])/self.extent[axis])
        self.constrain(p)


def reference(ctx,camera,width,device):
    cam=camera.resize(width,round(camera.height*width/camera.width))
    rgba_image=Image.open(ctx.stage('m2')/'object_rgba.png').convert('RGBA')
    mask_image=Image.open(ctx.stage('m2')/'object_mask.png').convert('L')
    if rgba_image.size!=(camera.width,camera.height) or mask_image.size!=rgba_image.size:
        raise ValueError('M2 reference pixels must match the saved Anchor camera; no implicit crop/warp')
    rgba=np.asarray(rgba_image.resize((cam.width,cam.height),Image.Resampling.LANCZOS),np.float32)/255
    mask=np.asarray(mask_image.resize((cam.width,cam.height),Image.Resampling.LANCZOS),np.float32)/255
    rgb=rgba[:,:,:3]*rgba[:,:,3:]+1-rgba[:,:,3:]
    if (mask>.5).sum()<20:raise ValueError('Empty M2 reference mask')
    distance=distance_transform_edt(mask<.5)/max(1,np.sqrt((mask>.5).sum()))
    to=lambda x:torch.as_tensor(x.copy(),device=device,dtype=torch.float32)
    return {'camera':cam,'mask':to(mask),'rgb':to(rgb.transpose(2,0,1)),'distance':to(distance)}


def image_metrics(pred,ref):
    a=pred['alpha'];t=ref['mask'];rgb=pred['rgb'];truth=ref['rgb']
    intersection=(a*t).sum();iou=intersection/(a.sum()+t.sum()-intersection).clamp_min(1)
    boundary=(a*ref['distance']).sum()/a.sum().clamp_min(1)
    # Global illumination mismatch cannot win the orientation search: compare
    # normalized spatial RGB structure plus weak chromaticity, not raw color.
    weight=(a*t).detach();den=weight.sum().clamp_min(1)
    x=F.avg_pool2d(rgb[None],5,1,2)[0];y=F.avg_pool2d(truth[None],5,1,2)[0]
    xm=(x*weight).sum((1,2),keepdim=True)/den;ym=(y*weight).sum((1,2),keepdim=True)/den
    xs=((x-xm).square()*weight).sum((1,2),keepdim=True)/den
    ys=((y-ym).square()*weight).sum((1,2),keepdim=True)/den
    normalized=((x-xm)/xs.clamp_min(.0025).sqrt()-(y-ym)/ys.clamp_min(.0025).sqrt()).abs().clamp_max(2)
    structure=(normalized*weight).sum()/(3*den*2)
    chroma=((x/(x.sum(0,keepdim=True)+.1)-y/(y.sum(0,keepdim=True)+.1)).abs()*weight).sum()/(3*den)
    appearance=.8*structure+.2*chroma
    binary=a>.5;target=t>.5
    hard=(binary&target).sum()/(binary|target).sum().clamp_min(1)
    mae=((rgb-truth).abs()*t).sum()/(3*t.sum().clamp_min(1))
    return {'soft_iou':iou,'mask_iou':hard,'appearance_loss':appearance,'chromaticity_loss':chroma,'foreground_rgb_mae':mae,'boundary_loss':boundary}


def evaluate(model,p,ref):
    pred=model.render(p,ref['camera']);m=image_metrics(pred,ref)
    center,volume,_,_=model.spatial(p);cfg=model.cfg
    # Optional illumination-normalized color cue disambiguates otherwise
    # similar cabinet sides without letting absolute HDRI brightness dominate.
    loss=cfg['silhouette_weight']*(1-m['soft_iou'])+cfg['rgb_weight']*m['appearance_loss']+cfg.get('chromaticity_weight',0.)*m['chromaticity_loss']+cfg['boundary_weight']*m['boundary_loss']+cfg['center_weight']*center+cfg['volume_weight']*volume
    m.update(center_penalty=center,volume_penalty=volume,objective=loss)
    return loss,m,pred


def numbers(m):return {k:float(v.detach()) for k,v in m.items()}


def pose_report(model,p):
    with torch.no_grad():
        scale,rot,shift,ry,offset=model.placement(p);_,_,pos,sigma=model.spatial(p)
        low=torch.quantile(pos-sigma,.002,dim=0);high=torch.quantile(pos+sigma,.998,dim=0)
        excess=(F.relu(model.low-low)+F.relu(high-model.high))/model.extent
        center=shift+rot@((model.tensor(model.bounds[0])+model.tensor(model.bounds[1]))/2)*scale
        # Autograd uses float32; export an exactly orthonormal float64 SO(3)
        # matrix so bandwise SH rotation does not acquire cross-band leakage.
        rotation=Rotation.from_matrix(rot.cpu().numpy().astype(np.float64)).as_matrix()
        matrix=np.eye(4);matrix[:3,:3]=float(scale)*rotation;matrix[:3,3]=shift.cpu()
        return {'scale':float(scale),'yaw_degrees':float(torch.rad2deg(p[0]))%360,
            'horizontal_offset_fraction':p[2:].tolist(),'rotation':rotation.tolist(),'translation':shift.tolist(),
            'object_to_world':matrix.tolist(),'world_to_object':np.linalg.inv(matrix).tolist(),
            'device_center_world':center.tolist(),'support_center_world':(model.center+model.axes@offset).tolist(),
            'support_gap':0.,'bounds_local_low':low.tolist(),'bounds_local_high':high.tolist(),
            'extent_to_generation_ratio':((high-low)/model.extent).tolist(),'max_volume_overflow_fraction':float(excess.max()),
            'parameters_radians_logscale_dx_dz':p.tolist()}


def fit_rasterized(ctx):
    from .pbr_gaussian_optimizer import render as exported_render
    c=gsle(ctx);cfg={**DEFAULTS,**ctx.cfg['m4']['integration'].get('registration',{})}
    out=ctx.stage('m4');folder=out/'alignment';folder.mkdir(exist_ok=True)
    source=ctx.stage('m3')/'object_refined_local.ply';source_sha=sha256(source)
    local=c.GaussianData.load(source);space=registration_space(ctx);model=Pose(c,local,space,cfg)
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    ref=reference(ctx,camera,cfg['search_resolution'],local.xyz.device)
    fine=reference(ctx,camera,cfg['refine_resolution'],local.xyz.device)
    seeds=[];start=time.monotonic()
    for yaw in np.arange(-180,180,cfg['yaw_step_degrees']):
        rotated=(model.probe-model.pivot)@yaw_matrix(model.tensor(np.deg2rad(yaw))).T
        sides=(rotated.max(0).values-rotated.min(0).values).clamp_min(1e-6)
        scale=float((model.extent/sides).min())
        for fraction in cfg['initial_size_fractions']:
            p=model.tensor([np.deg2rad(yaw),math.log(scale*fraction),0,0]);model.constrain(p)
            with torch.no_grad():_,m,_=evaluate(model,p,ref)
            seeds.append({'parameters':p.tolist(),**numbers(m)})
    historical=ctx.cfg['m4']['integration'].get('existing_transform')
    baseline=None
    if historical:
        original=read_json(ctx.source(historical));rot=model.tensor(original['rotation']);scale=float(original['scale'])
        ry=model.axes.T@rot;yaw=torch.atan2(ry[0,2],ry[0,0])
        offset=model.axes.T@(model.tensor(original['translation'])-model.center)+scale*ry@model.pivot
        d=(offset-model.support)/model.extent
        p=model.tensor([float(yaw),math.log(scale),float(d[0]),float(d[2])])
        with torch.no_grad():_,m,_=evaluate(model,p,ref)
        seeds.append({'parameters':p.tolist(),**numbers(m),'source':'historical transform (support reanchored)'})
        old=c.GaussianData.load(Path(ctx.source(historical)).parent/'object_world.ply')
        with torch.no_grad():
            pred=exported_render(c,fine['camera'],old);baseline=numbers(image_metrics(pred,fine))
            save_render(folder,'old',pred,fine,c)
        del old
    seeds.sort(key=lambda r:r['objective']);starts=[]
    for row in seeds:
        yaw=row['parameters'][0]
        if all(abs((yaw-s['parameters'][0]+math.pi)%(2*math.pi)-math.pi)>=math.radians(30) for s in starts):starts.append(row)
        if len(starts)==cfg['refine_candidates']:break
    candidates=[];trace=[]
    for index,row in enumerate(starts):
        p=torch.nn.Parameter(model.tensor(row['parameters']));optimizer=torch.optim.Adam([p],lr=cfg['learning_rate'])
        best=(float('inf'),p.detach().clone())
        for step in range(cfg['coarse_steps']):
            optimizer.zero_grad();loss,m,_=evaluate(model,p,ref)
            if float(loss.detach())<best[0]:best=(float(loss.detach()),p.detach().clone())
            loss.backward()
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Nonfinite pose gradient')
            torch.nn.utils.clip_grad_norm_([p],1.);optimizer.step();model.constrain(p)
        p=best[1]
        with torch.no_grad():_,m,_=evaluate(model,p,fine)
        item={'candidate':index,'parameters':p.tolist(),**numbers(m),**pose_report(model,p)}
        candidates.append(item);print('ALIGNMENT_CANDIDATE',index,item['mask_iou'],item['objective'],item['yaw_degrees'],flush=True)
        write_json(folder/'progress.json',{'phase':'candidates','complete':index+1,'total':len(starts)})
    candidates.sort(key=lambda r:r['objective'])
    # Optional diversity after optimization: different initial yaws can converge
    # to the same pose and otherwise consume both final refinement slots.
    final_starts=[]
    for row in candidates:
        if all(abs((row['yaw_degrees']-x['yaw_degrees']+180)%360-180)>=cfg.get('fine_min_yaw_separation_degrees',0) for x in final_starts):
            final_starts.append(row)
        if len(final_starts)==2:break
    # Refine the highest-scoring retained candidates at the final evaluation size.
    finalists=[]
    for row in final_starts:
        p=torch.nn.Parameter(model.tensor(row['parameters']));optimizer=torch.optim.Adam([p],lr=cfg['learning_rate']/3)
        best=(float('inf'),p.detach().clone())
        for step in range(cfg['fine_steps']):
            optimizer.zero_grad();loss,m,_=evaluate(model,p,fine)
            if float(loss.detach())<best[0]:best=(float(loss.detach()),p.detach().clone())
            if step%30==0:trace.append({'candidate':row['candidate'],'step':step,**numbers(m)})
            loss.backward();torch.nn.utils.clip_grad_norm_([p],1.);optimizer.step();model.constrain(p)
        p=best[1]
        with torch.no_grad():_,m,_=evaluate(model,p,fine)
        finalists.append({'parameters':p.tolist(),**numbers(m),**pose_report(model,p)})
    finalists.sort(key=lambda r:r['objective']);selected=finalists[0];p=model.tensor(selected['parameters'])
    constrained=[]
    if selected['max_volume_overflow_fraction']>cfg['max_volume_fraction']:
        # Same image losses and thresholds; explicitly enforce the space
        # constraint rather than accepting a better-looking invalid placement.
        for row in finalists:
            q=torch.nn.Parameter(model.tensor(row['parameters']));model.project_volume(q)
            optimizer=torch.optim.Adam([q],lr=cfg['learning_rate']/3);best=(float('inf'),q.detach().clone())
            for step in range(cfg['fine_steps']):
                optimizer.zero_grad();loss,m,_=evaluate(model,q,fine)
                if float(loss.detach())<best[0]:best=(float(loss.detach()),q.detach().clone())
                loss.backward();torch.nn.utils.clip_grad_norm_([q],1.);optimizer.step();model.project_volume(q)
            q=best[1]
            with torch.no_grad():_,m,_=evaluate(model,q,fine)
            constrained.append({'parameters':q.tolist(),**numbers(m),**pose_report(model,q)})
        valid=[r for r in constrained if r['max_volume_overflow_fraction']<=cfg['max_volume_fraction'] and r['mask_iou']>=cfg['min_silhouette_iou']]
        if valid:selected=min(valid,key=lambda r:r['objective']);p=model.tensor(selected['parameters'])
        write_json(folder/'constrained_refinement.json',{'gate_unchanged':True,'input_finalists':finalists,'candidates':constrained,'selected_feasible':bool(valid)})
    write_json(folder/'search.json',{'settings':cfg,'coarse_search':seeds,'refined_candidates':candidates,'finalists':finalists,'constrained_candidates':constrained,'selected':selected,'history':trace})
    if selected['mask_iou']<cfg['min_silhouette_iou'] or selected['max_volume_overflow_fraction']>cfg['max_volume_fraction']:
        raise ValueError('Registration failed silhouette/space checks; inspect alignment/search.json')
    vertices=read_vertices(source);pose=pose_report(model,p)
    world=transform_vertices(vertices,pose['scale'],np.asarray(pose['rotation']),np.asarray(pose['translation']))
    save_vertices(out/'object_world.ply',world)
    write_json(out/'transform.json',{**pose,'method':'rasterized_reference_v1','camera':camera.record(),
        'generation_frame':space['frame'],'registration_settings':cfg,'reference_mask':str(ctx.stage('m2')/'object_mask.png'),
        'source_local':str(source),'source_local_sha256':source_sha,'metrics':selected,
        'floor_local_y':space['bottom_local_y'],'parameters_optimized':['yaw','uniform_scale','horizontal_x','horizontal_z']})
    exported=c.GaussianData.load(out/'object_world.ply')
    with torch.no_grad():
        direct=model.render(p,fine['camera']);actual=exported_render(c,fine['camera'],exported)
        equivalence=float((direct['rgb']-actual['rgb']).abs().mean())
        if equivalence>2e-4:raise ValueError('Exported SH/covariance transform disagrees with pose renderer')
        save_render(folder,'new',actual,fine,c)
        full=reference(ctx,camera,camera.width,local.xyz.device)
        save_render(folder,'anchor',exported_render(c,camera,exported),full,c)
    assert sha256(source)==source_sha
    report={'baseline':baseline,'selected':selected,'space':space,'camera':camera.record(),
        'camera_convention':'COLMAP world-to-camera +Z forward; saved M2 crop/resize intrinsics retained',
        'local_convention':'M3 GLB Y-up (Blender axis conversion already undone in M3)',
        'source_sha256_unchanged':True,'gaussian_count_unchanged':len(world)==len(vertices),
        'opacity_bitwise_unchanged':world['opacity'].tobytes()==vertices['opacity'].tobytes(),
        'export_rgb_equivalence_mae':equivalence,'seconds':time.monotonic()-start,'no_device_training':True}
    write_json(folder/'report.json',report)
    print('ALIGNMENT_COMPLETE',baseline,selected,flush=True)
    del exported,local,model;torch.cuda.empty_cache()
    return out/'object_world.ply'


def save_render(folder,name,pred,ref,common):
    common.save_image(folder/(name+'_rgb.png'),pred['rgb'])
    alpha=pred['alpha'].detach().cpu().numpy();truth=ref['mask'].cpu().numpy()
    Image.fromarray(np.round(alpha.clip(0,1)*255).astype('uint8')).save(folder/(name+'_alpha.png'))
    np.save(folder/(name+'_depth_z.npy'),pred['depth_z'].detach().cpu().numpy())
    overlay=np.stack((alpha,truth,np.zeros_like(alpha)),-1)
    Image.fromarray(np.round(overlay.clip(0,1)*255).astype('uint8')).save(folder/(name+'_overlay.png'))
    common.save_image(folder/'reference_white.png',ref['rgb'])
