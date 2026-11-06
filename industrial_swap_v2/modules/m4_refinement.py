"""Scene-aware text_only DI-SDS, initialized from TRELLIS world-space Gaussians."""
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ..core.context import Context, arguments, gsle, read_json, write_json, sha256
from ..core.scene import compose, read_vertices
from ..core.rendering import render_outputs
from ..core.m3_composition import effective_background


class AppearanceGaussians(torch.nn.Module):
    """Geometry buffers are immutable; only SH0 RGB and opacity are optimized."""
    def __init__(self, data, common):
        super().__init__(); self.common=common
        for name in ('xyz','log_scale','rotation'):
            self.register_buffer(name,getattr(data,name).detach().clone())
        self.register_buffer('sh_rest',data.sh[:,:,1:].detach().clone())
        color=(data.sh[:,:,0]*0.28209479177387814+.5).clamp(1e-5,1-1e-5)
        self.color_logit=torch.nn.Parameter(torch.logit(color))
        self.opacity_logit=torch.nn.Parameter(data.opacity_logit.detach().clone())

    def data(self):
        dc=self.common.RGB2SH(self.color_logit.sigmoid())[:,:,None]
        return self.common.GaussianData(self.xyz,self.log_scale,self.rotation,self.opacity_logit,
                                       torch.cat((dc,self.sh_rest),2))


def image_tensor(path,mode):
    with Image.open(path) as image: array=np.asarray(image.convert(mode)).copy()/255
    if array.ndim==2: array=array[:,:,None]
    return torch.tensor(array.transpose(2,0,1),dtype=torch.float32,device='cuda')


def checkpoint(path,obj,optimizer,step,fingerprint):
    temporary=path.with_suffix('.tmp')
    torch.save({'state':obj.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
        'fingerprint':fingerprint,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
        'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}},temporary)
    temporary.replace(path)


def run(ctx):
    cfg=ctx.cfg['m4']; out=ctx.stage('m4')
    if not cfg['freeze_background'] or not cfg['preserve_geometry']:
        raise ValueError('This v2 preset requires frozen background and preserved TRELLIS geometry')
    assert cfg['unconditional_conditioning']=='text_only'
    assert cfg['resolution']==512 and 1<=cfg['microbatch']<=cfg['batch']
    source=ctx.path(cfg['initialize_from'])
    if not source.is_file(): raise FileNotFoundError('M3 world-space Gaussian asset is required')
    if cfg.get('local_background',{}).get('enabled',False):
        from ..core.local_background_optimizer import optimize_background
        background_path=optimize_background(ctx)
    else:
        background_path=effective_background(ctx)
    fingerprint_data={'m4':cfg,'source':sha256(source),
        'reference':sha256(ctx.stage('m2')/'object_rgba.png'),
        'camera':read_json(ctx.stage('m2')/'anchor_camera.json'),
        'text':ctx.cfg['m2']['replacement_text']}
    if background_path.resolve()!=(ctx.stage('m1')/'background_gaussians.ply').resolve():
        fingerprint_data['background']=sha256(background_path)
    fingerprint=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
    teacher_cfg=read_json(ctx.source('gsle/config.json'))
    teacher_cfg['replacement_text']=ctx.cfg['m2']['replacement_text']
    teacher_cfg['refine'].update(cfg)
    teacher_cfg['assumptions'].append('V2: initialize from TRELLIS.2 surface Gaussians; preserve geometry, optimize RGB/opacity; text_only CFG.')
    teacher_path=out/'effective_teacher_config.json'; write_json(teacher_path,teacher_cfg)
    os.environ['GSLE_CONFIG']=str(teacher_path)
    c=gsle(ctx)
    from diffusion import DepthInpaintingGuidance
    from camera_sampling import sample_views
    if torch.cuda.mem_get_info()[0]/2**30<cfg['minimum_free_gpu_gib']:
        raise RuntimeError('Insufficient free GPU memory; shared jobs are untouched')
    if cfg.get('memory_budget_gib'):
        torch.cuda.set_per_process_memory_fraction(cfg['memory_budget_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    write_json(out/'resource_policy.json',{'physical_gpu':torch.cuda.get_device_name(0),
        'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'available_gib_at_start':torch.cuda.mem_get_info()[0]/2**30,
        'torch_allocator_budget_gib':cfg.get('memory_budget_gib'),'microbatch':cfg['microbatch'],'effective_batch':cfg['batch']})
    seed=ctx.cfg['project']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data=c.GaussianData.load(source); obj=AppearanceGaussians(data,c); del data
    optimizer=torch.optim.Adam([{'params':[obj.color_logit],'lr':cfg['color_lr']},
        {'params':[obj.opacity_logit],'lr':cfg['opacity_lr']}],eps=1e-15)
    last=out/'checkpoint_latest.pth'; first=0; saved=None
    if last.exists():
        saved=torch.load(last,map_location='cuda',weights_only=False)
        assert saved['fingerprint']==fingerprint, 'Refinement checkpoint input/config mismatch'
        obj.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer']); first=saved['step']
    background=c.GaussianData.load(background_path)
    assert not any(getattr(background,n).requires_grad for n in ('xyz','log_scale','rotation','opacity_logit','sh'))
    reference=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    region=read_json(ctx.stage('m1')/'edit_region.json')
    center=np.asarray(region['center']); up=np.asarray(region['up'])
    toward=reference.center-center; radius=np.linalg.norm(toward)
    horizontal=c.normalized(toward-up*np.dot(toward,up)); right=c.normalized(np.cross(up,horizontal))
    object_scale=float(region.get('object_scale',np.linalg.norm(np.asarray(region['bbox_corners'])-center,axis=1).max()))
    reference_rgb=image_tensor(ctx.stage('m2')/'object_white.png','RGB')
    reference_mask=image_tensor(ctx.stage('m2')/'object_mask.png','L')
    resolution=cfg['resolution']
    gt_rgb=F.interpolate(reference_rgb[None],(resolution,resolution),mode='bilinear',align_corners=False)[0]
    gt_mask=F.interpolate(reference_mask[None],(resolution,resolution),mode='nearest')[0]
    guide=DepthInpaintingGuidance() if first<cfg['iterations'] else None
    if saved:
        rng=saved['rng']; random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])
    write_json(out/'effective_config.json',{'config':ctx.cfg,'fingerprint':fingerprint,
        'source':str(source),'background_source':str(background_path),'background_sha256':sha256(background_path),
        'diffusion_mask_source':cfg.get('diffusion_mask_source','edit_bbox'),
        'optimized_parameters':['color_logit','opacity_logit'],
        'frozen_parameters':['xyz','log_scale','rotation','background','diffusion_teachers'],
        'reference_loss':'isolated white-background object RGB and SAM alpha; joint scene RGB/depth/mask for DI-SDS',
        'checkpoint_interval':cfg['checkpoint_every']})
    start=time.time()
    for step in range(first,cfg['iterations']):
        tick=time.time(); optimizer.zero_grad(set_to_none=True)
        result=c.render(reference.resize(resolution,resolution),obj.data(),background=torch.ones(3,device='cuda'))
        rgb_loss=F.mse_loss(result['rgb'],gt_rgb); mask_loss=F.mse_loss(result['alpha'],gt_mask)
        values={'rgb_mse':float(rgb_loss.detach()),'mask_mse':float(mask_loss.detach())}
        (cfg['rgb_weight']*rgb_loss+cfg['mask_weight']*mask_loss).backward()
        del result,rgb_loss,mask_loss
        views=sample_views(teacher_cfg['camera_sampling'],cfg['batch'],step,resolution,
            center,up,horizontal,right,radius,object_scale)
        score=0.; diagnostics=[]
        for begin in range(0,len(views),cfg['microbatch']):
            rgbs=[]; depths=[]; masks=[]; prompts=[]
            for sample in views[begin:begin+cfg['microbatch']]:
                camera=sample['camera']; result=c.render_joint(camera,background,obj.data())
                rgbs.append(result['rgb'].clamp(0,1)); depths.append(c.depth_condition(result['depth'],result['alpha']))
                if cfg.get('diffusion_mask_source')=='object_alpha':
                    from ..core.visibility_loss import object_projection_mask
                    mask,_=object_projection_mask(c,camera,obj.data(),dilation_fraction=cfg.get('object_mask_dilation_fraction',.015))
                    masks.append(mask)
                    if step==0:
                        c.save_image(out/f'diffusion_masks/view_{begin+len(masks):02d}.png',mask)
                else:
                    masks.append(torch.tensor(c.bbox_mask(camera,np.asarray(region['bbox_corners'])),device='cuda',dtype=torch.float32)[None])
                az=abs(sample['azimuth']); el=sample['elevation']
                view='overhead view' if el>60 else 'front view' if az<45 else 'back view' if az>135 else 'side view'
                prompts.append(ctx.cfg['m2']['replacement_text']+', '+view)
            loss,diagnostic=guide.loss_batch(torch.stack(rgbs),torch.stack(depths),torch.stack(masks),prompts)
            (loss*(cfg['sds_weight']/cfg['batch'])).backward()
            score+=float(loss.detach())/cfg['batch']; diagnostics.extend(diagnostic)
            del result,loss,rgbs,depths,masks
        gradients={name:float(parameter.grad.norm()) for name,parameter in obj.named_parameters()}
        if not all(math.isfinite(v) for v in gradients.values()): raise FloatingPointError('Non-finite refinement gradient')
        optimizer.step()
        record={'step':step+1,'seconds':time.time()-tick,'elapsed_this_attempt':time.time()-start,
            **values,'sds_loss':score,'gradient_norms':gradients,'diffusion':diagnostics,
            'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,'points':len(obj.xyz)}
        with (out/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        if step%10==0: print({k:v for k,v in record.items() if k!='diffusion'},flush=True)
        if step==0 or (step+1)%cfg['checkpoint_every']==0 or step+1==cfg['iterations']:
            checkpoint(last,obj,optimizer,step+1,fingerprint)
            with torch.no_grad():
                c.save_image(out/f'previews/anchor_{step+1:06d}.png',c.render_joint(reference,background,obj.data(),aux=False)['rgb'])
            write_json(ctx.records/'m4_progress.json',record)
    checkpoint(out/'final_checkpoint.pth',obj,optimizer,cfg['iterations'],fingerprint)
    obj.data().save(out/'object_refined.ply')
    before,after=read_vertices(source),read_vertices(out/'object_refined.ply')
    for name in ['x','y','z']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]:
        assert np.array_equal(before[name],after[name]), ('Geometry changed',name)
    compose(background_path,out/'object_refined.ply',out/'scene_final.ply',out/'composition.json')
    write_json(out/'summary.json',{'iterations':cfg['iterations'],'points':len(obj.xyz),
        'geometry_bitwise_preserved':True,'background_bitwise_preserved':True,'condition':'text_only',
        'background_source':str(background_path),'background_sha256':sha256(background_path),
        'diffusion_mask_source':cfg.get('diffusion_mask_source','edit_bbox'),
        'initialized_from_trellis':True,'zero123_used':False})
    del guide,obj,optimizer,background; torch.cuda.empty_cache()
    render_outputs(ctx,'m4',out/'object_refined.ply',background_path=background_path)
    ctx.complete('m4',[out/n for n in ('object_refined.ply','scene_final.ply','final_checkpoint.pth',
        'renders/anchor_after.png','renders/trajectory_before_after.mp4','renders/orbit_before_after.mp4')],iterations=cfg['iterations'])


if __name__=='__main__': run(Context(arguments().config))
