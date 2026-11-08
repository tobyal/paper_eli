"""Explicit RGB + camera-Z supervision of a GLB-initialized Gaussian object.

No image diffusion, mask loss, density control, or background optimization.
All validation cameras are disjoint from gradient-producing training cameras.
"""
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from .context import gsle,read_json,write_json,sha256
from .pbr_math import ray_distance_to_z,read_depth_exr


def finalize_dataset(folder):
    os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
    import cv2
    manifest=read_json(folder/'cameras.json');rows=[]
    calibration=read_json(folder/'ray_depth_calibration.json')
    if not calibration['passed']: raise ValueError('Depth calibration must pass before training')
    cache=folder/'cache';cache.mkdir(exist_ok=True)
    for record in manifest['views']:
        rgba=np.asarray(Image.open(folder/record['rgba']).convert('RGBA'),dtype=np.float32)/255
        alpha=rgba[:,:,3];rgb=rgba[:,:,:3]*alpha[:,:,None]+1-alpha[:,:,None]
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(folder/record['rgb'])
        # PNG alpha is linear coverage. Preserve it instead of a display-encoded mask.
        Image.fromarray(np.round(alpha*255).astype(np.uint8)).save(folder/record['mask'])
        raw=read_depth_exr(folder/record['depth'])
        valid=np.isfinite(raw)&(raw>0)&(raw<manifest['camera_distance']+manifest['radius']*2)&(alpha>.99)
        depth=np.where(valid,raw,0).astype(np.float32)
        if calibration['verified_definition']=='ray_distance': depth=ray_distance_to_z(depth,record)
        if valid.sum()<100 or not np.isfinite(depth).all(): raise ValueError('Invalid foreground depth')
        ys,xs=np.where(alpha>.01)
        if xs.min()<2 or ys.min()<2 or xs.max()>=record['width']-2 or ys.max()>=record['height']-2:
            raise ValueError('Camera clips the device')
        np.savez_compressed(cache/(record['name']+'.npz'),rgb=rgb.astype(np.float32),alpha=alpha,
                            depth_z=depth,valid_depth=valid)
        rows.append({'name':record['name'],'split':record['split'],'foreground_pixels':int((alpha>.5).sum()),
                     'valid_depth_pixels':int(valid.sum()),'depth_z_min':float(depth[valid].min()),
                     'depth_z_max':float(depth[valid].max()),
                     'files':{key:sha256(folder/record[key]) for key in ('rgb','rgba','depth','mask')}})
    train={r['name'] for r in rows if r['split']=='train'}
    val={r['name'] for r in rows if r['split']=='validation'}
    assert train and val and train.isdisjoint(val)
    write_json(folder/'dataset_validation.json',{'views':rows,'train_count':len(train),'validation_count':len(val),
        'depth':calibration['verified_definition']+' EXR, converted to camera Z only if necessary',
        'rgb':'saved display-sRGB RGBA composited onto display-white; no exposure fitting',
        'mask_role':'valid depth/inspection only, no mask loss','all_objects_fully_in_frame':True})
    return manifest


class TrainableObject(torch.nn.Module):
    def __init__(self,data,common,cfg,size):
        super().__init__();self.common=common;self.cfg=cfg;self.size=size
        for key in ('xyz','log_scale','rotation','opacity_logit','sh'):
            value=getattr(data,key).detach().clone()
            if key=='rotation': self.register_buffer(key,value)
            elif key=='sh':
                self.dc=torch.nn.Parameter(value[:,:,:1].clone());self.rest=torch.nn.Parameter(value[:,:,1:].clone())
            else: setattr(self,key,torch.nn.Parameter(value))
        self.register_buffer('initial_xyz',data.xyz.detach().clone())
        self.register_buffer('initial_scale',data.log_scale.detach().clone())

    def data(self):
        return self.common.GaussianData(self.xyz,self.log_scale,self.rotation,self.opacity_logit,
                                       torch.cat((self.dc,self.rest),dim=2))

    def optimizer(self):
        cfg=self.cfg
        groups=[{'params':[getattr(self,name)],'name':name,'lr':rate} for name,rate in (
            ('dc',cfg['sh_dc_lr']),('rest',cfg['sh_rest_lr']),('opacity_logit',cfg['opacity_lr']),
            ('log_scale',cfg['scale_lr']),('xyz',cfg['xyz_lr_fraction']*self.size))]
        return torch.optim.Adam(groups,lr=0,eps=1e-15)

    @torch.no_grad()
    def constrain(self):
        self.xyz.copy_(self.initial_xyz+(self.xyz-self.initial_xyz).clamp(
            -self.cfg['max_xyz_offset_fraction']*self.size,self.cfg['max_xyz_offset_fraction']*self.size))
        self.log_scale.copy_(torch.maximum(self.initial_scale+math.log(self.cfg['min_scale_ratio']),
            torch.minimum(self.initial_scale+math.log(self.cfg['max_scale_ratio']),self.log_scale)))
        self.opacity_logit.clamp_(math.log(.01/.99),math.log(.995/.005))


def render(common,camera,data):
    rgb=common.raster(camera,data,data.colors(camera.center),background=torch.ones(3,device='cuda'))
    # Unlike the historical scene renderer, this is camera Z, alpha-normalized,
    # and stays differentiable through geometry, covariance, and opacity.
    R=torch.as_tensor(camera.R,device='cuda');t=torch.as_tensor(camera.t,device='cuda')
    z=(data.xyz@R[2]+t[2]);packed=common.raster(camera,data,torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),1))
    alpha=packed[1];depth=packed[0]/alpha.clamp_min(1e-6)
    return {'rgb':rgb,'depth_z':depth,'alpha':alpha}


def losses(pred,view,size):
    rgb=(pred['rgb']-view['rgb']).abs().mean()
    valid=view['valid_depth']
    depth=(pred['depth_z'][valid]-view['depth_z'][valid]).abs().mean()/size
    return rgb,depth


@torch.no_grad()
def evaluate(model,views,size,cfg):
    rows=[]
    for view in views:
        pred=render(model.common,view['camera'],model.data());rgb,depth=losses(pred,view,size)
        fg=view['alpha']>.5;mask=pred['alpha']>.5
        mse=(pred['rgb']-view['rgb']).square()
        rows.append({'name':view['name'],'split':view['split'],'rgb_l1':rgb.item(),
            'psnr':float(-10*torch.log10(mse.mean().clamp_min(1e-12))),
            'foreground_psnr':float(-10*torch.log10(mse[:,fg].mean().clamp_min(1e-12))),
            'depth_mae_relative':depth.item(),'depth_mae':depth.item()*size,
            'silhouette_iou':float((fg&mask).sum()/(fg|mask).sum().clamp_min(1))})
    result={'views':rows}
    for split in ('train','validation'):
        subset=[r for r in rows if r['split']==split]
        result[split]={k:float(np.mean([r[k] for r in subset])) for k in
            ('rgb_l1','psnr','foreground_psnr','depth_mae_relative','depth_mae','silhouette_iou')}
        result[split]['objective']=cfg['rgb_weight']*result[split]['rgb_l1']+cfg['depth_weight']*result[split]['depth_mae_relative']
    return result


def optimize(ctx):
    c=gsle(ctx);out=ctx.stage('m3');cfg=ctx.cfg['m3']['pbr']['training']
    if cfg['sh_degree']!=3: raise ValueError('The scene PLY schema requires SH degree 3')
    if torch.cuda.mem_get_info()[0]/2**30<cfg['minimum_free_gpu_gib']:
        raise RuntimeError('Insufficient free GPU memory; other jobs are untouched')
    manifest=finalize_dataset(out/'pbr_views');size=float(np.linalg.norm(np.array(manifest['bbox_high'])-manifest['bbox_low']))
    initial=c.GaussianData.load(out/'object_local.ply');model=TrainableObject(initial,c,cfg,size)
    optimizer=model.optimizer();rng=np.random.default_rng(ctx.cfg['project']['seed']);views=[]
    for record in manifest['views']:
        values=np.load(out/'pbr_views/cache'/(record['name']+'.npz'))
        view={'name':record['name'],'split':record['split'],'camera':c.Camera.from_record(record)}
        for key in ('rgb','alpha','depth_z','valid_depth'):
            value=values[key];value=value.transpose(2,0,1) if key=='rgb' else value
            view[key]=torch.as_tensor(value.copy(),device='cuda')
        views.append(view)
    train=[v for v in views if v['split']=='train'];history=[];start=time.monotonic()
    checkpoint=out/'checkpoints';checkpoint.mkdir(exist_ok=True)
    baseline=evaluate(model,views,size,cfg);best=baseline['validation']['objective'];best_step=0
    write_json(out/'baseline_metrics.json',baseline)
    # Catch coordinate/depth mistakes before any optimization can hide them.
    if baseline['validation']['silhouette_iou']<.85 or baseline['validation']['depth_mae_relative']>.04:
        raise ValueError('GLB/Gaussian calibration check failed; inspect baseline_metrics.json')
    signature={'initial_sha256':sha256(out/'object_local.ply'),'cameras_sha256':sha256(out/'pbr_views/cameras.json'),
               'config':cfg,'seed':ctx.cfg['project']['seed']}
    def save(step,path,metrics):
        torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'rng':rng.bit_generator.state,'signature':signature,'metrics':metrics},path)
    latest=checkpoint/'latest.pth';first=1
    if not latest.exists():
        save(0,checkpoint/'best.pth',baseline);model.data().save(out/'object_refined_local.ply')
    if latest.exists():
        saved=torch.load(latest,map_location='cuda',weights_only=False)
        if saved['signature']!=signature: raise ValueError('Training resume signature mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];first=saved['step']+1
        old=read_json(out/'training_progress.json');best=old['best_objective'];best_step=old['best_step']
        history=read_json(out/'evaluations.json')
    trace=(out/'training.jsonl').open('a',buffering=1)
    try:
        import json
        for step in range(first,cfg['iterations']+1):
            view=train[int(rng.integers(len(train)))];optimizer.zero_grad(set_to_none=True)
            prediction=render(c,view['camera'],model.data());rgb,depth=losses(prediction,view,size)
            loss=cfg['rgb_weight']*rgb+cfg['depth_weight']*depth
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            for name,parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError('Invalid or disconnected gradient: '+name)
            optimizer.step();model.constrain()
            trace.write(json.dumps({'step':step,'camera':view['name'],'rgb_l1':rgb.item(),
                'depth_relative_l1':depth.item(),'loss':loss.item(),'seconds':time.monotonic()-start})+'\n')
            if step%cfg['evaluate_every']==0 or step==cfg['iterations']:
                metrics=evaluate(model,views,size,cfg);score=metrics['validation']['objective']
                if score<best:
                    best=score;best_step=step;save(step,checkpoint/'best.pth',metrics)
                    model.data().save(out/'object_refined_local.ply')
                history.append({'step':step,**metrics})
                write_json(out/'evaluations.json',history)
                save(step,latest,metrics)
                write_json(out/'training_progress.json',{'step':step,'best_step':best_step,'best_objective':best,
                    'seconds':time.monotonic()-start,'validation':metrics['validation']})
                print('PBR_TRAIN',step,'best',best_step,metrics['validation'],flush=True)
            if step%cfg['save_every']==0:
                model.data().save(checkpoint/f'object_step{step:06d}.ply')
    finally: trace.close()
    saved=torch.load(checkpoint/'best.pth',map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
    best_metrics=evaluate(model,views,size,cfg)
    images=out/'renders';images.mkdir(exist_ok=True)
    boards=[]
    with torch.no_grad():
        for view in views:
            before=render(c,view['camera'],initial);after=render(c,view['camera'],model.data())
            c.save_image(images/(view['name']+'_initial.png'),before['rgb'])
            c.save_image(images/(view['name']+'_refined.png'),after['rgb'])
            panel=torch.cat((view['rgb'],before['rgb'],after['rgb']),dim=2)
            c.save_image(images/(view['name']+'_comparison.png'),panel)
            np.savez_compressed(images/(view['name']+'_depth.npz'),reference=view['depth_z'].cpu().numpy(),
                initial=before['depth_z'].cpu().numpy(),refined=after['depth_z'].cpu().numpy())
            if view['split']=='validation': boards.append(images/(view['name']+'_comparison.png'))
    overview=Image.new('RGB',(1536,560*len(boards)),'white');draw=ImageDraw.Draw(overview)
    for row,path in enumerate(boards):
        draw.text((10,row*560+8),path.stem+' | PBR target | initial GS | optimized GS',fill='black')
        overview.paste(Image.open(path).convert('RGB'),(0,row*560+40))
    overview.save(images/'validation_comparison.jpg',quality=93)
    from .rendering import VideoWriter
    for elevation_index in range(len(ctx.cfg['m3']['pbr']['render']['elevations_degrees'])):
        selected=views[elevation_index*12:(elevation_index+1)*12]
        with VideoWriter(images/f'pbr_gaussian_orbit_{elevation_index}.mp4',3) as video:
            for view in selected:
                video.append_data(np.asarray(Image.open(images/(view['name']+'_comparison.png')).convert('RGB')))
    report={'status':'completed','method':'pbr_multiview_gaussian','iterations':cfg['iterations'],
        'best_step':int(saved['step']),'baseline':baseline,'best':best_metrics,'gaussian_count':len(initial.xyz),
        'seconds':time.monotonic()-start,'losses':['display_srgb_l1','valid_camera_z_l1_normalized_by_object_diagonal'],
        'mask_loss':False,'optimized_parameters':['SH degree 3','opacity','scale','bounded xyz'],
        'rotation_frozen':True,'topology_frozen':True,'train_views':len(train),'validation_views':len(views)-len(train),
        'object_refined_sha256':sha256(out/'object_refined_local.ply'),
        'limitation':'SH degree 3 approximates appearance under one fixed HDRI; not a relightable PBR material model.'}
    write_json(out/'pbr_training_summary.json',report)
    del model,initial,optimizer,views;torch.cuda.empty_cache()
    return report
