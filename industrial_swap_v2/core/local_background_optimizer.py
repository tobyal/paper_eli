"""M4: dual-mask local background SH0/opacity optimization, fixed device/topology."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .context import gsle, read_json, write_json, sha256
from .scene import read_vertices, save_vertices, compose
from .multiview_masks import ensure_m4_supervision, UNION_SCHEME
from .visibility_loss import masked_l1, visibility_loss, object_contribution
from .scene_integration import object_path as integrated_object

C0=0.28209479177387814
DEFAULTS={'enabled':True,'iterations':250,'color_lr':.004,'opacity_lr':.06,
    'rgb_weight':1.,'visibility_weight':1.,'opacity_preserve_weight':.05,
    'color_preserve_weight':.02,'max_rgb_change':.08,'max_opacity_change':.08,
    'occlusion_min_alpha':.005,'warmup_steps':20,'evaluate_every':50,
    'holdout_max_relative_degradation':.01,'occlusion_regularization':.002,
    'occlusion_environment_guard_weight':.5,'legal_foreground_guard_weight':2.,
    'max_outside_union_mae':.005,'max_legal_foreground_mae':.005}


def optimizer_config(values=None):
    values=dict(values or {})
    # These knobs had no effect in the previous union implementation.
    for key in ('outside_preserve_weight','candidate_max_opacity_change'): values.pop(key,None)
    cfg={**DEFAULTS,**values}
    if set(cfg)-set(DEFAULTS): raise ValueError('Unknown local background configuration keys')
    if not cfg['enabled'] or type(cfg['iterations']) is not int or not 200<=cfg['iterations']<=10000:
        raise ValueError('Official M4 requires 200–10000 local background steps')
    if cfg['evaluate_every']<1: raise ValueError('Evaluation interval must be positive')
    return cfg


class PermissionedBackground(torch.nn.Module):
    """One physical opacity per splat, independent loss-specific gradient gates."""
    def __init__(self,original,rgb_indices,occlusion_indices,common,cfg):
        super().__init__(); self.original=original; self.common=common; self.cfg=cfg
        device=original.xyz.device; indices=np.union1d(rgb_indices,occlusion_indices)
        self.register_buffer('indices',torch.as_tensor(indices,device=device,dtype=torch.long))
        self.register_buffer('rgb_indices',torch.as_tensor(rgb_indices,device=device,dtype=torch.long))
        self.register_buffer('rgb_permission',torch.tensor(np.isin(indices,rgb_indices),device=device)[:,None])
        self.register_buffer('occlusion_permission',torch.tensor(np.isin(indices,occlusion_indices),device=device)[:,None])
        logits=original.opacity_logit[self.indices].detach().clone()
        self.register_buffer('initial_logits',logits); self.register_buffer('initial_alpha',logits.sigmoid())
        self.opacity_delta=torch.nn.Parameter(torch.zeros_like(logits))
        self.dc_delta=torch.nn.Parameter(torch.zeros((len(rgb_indices),3),device=device))
        low=(self.initial_alpha-cfg['max_opacity_change']).clamp(1e-6,1-1e-6)
        high=(self.initial_alpha+cfg['max_opacity_change']).clamp(1e-6,1-1e-6)
        low=torch.where(self.occlusion_permission,torch.minimum(self.initial_alpha,torch.full_like(low,cfg['occlusion_min_alpha'])),low)
        high=torch.where(self.occlusion_permission,self.initial_alpha,high)
        self.register_buffer('lower',torch.minimum(torch.logit(low)-logits,torch.zeros_like(logits)))
        self.register_buffer('upper',torch.maximum(torch.logit(high)-logits,torch.zeros_like(logits)))

    def data(self,mode='all'):
        delta=self.opacity_delta
        if mode!='all':
            permission=self.rgb_permission if mode=='rgb' else self.occlusion_permission
            delta=torch.where(permission,delta,delta.detach())
        b=self.original
        opacity=b.opacity_logit.index_copy(0,self.indices,self.initial_logits+delta)
        return self.common.GaussianData(b.xyz,b.log_scale,b.rotation,opacity,b.sh)

    def render(self,camera,mode='all'):
        b=self.original
        with torch.no_grad():
            direction=F.normalize(b.xyz-torch.as_tensor(camera.center,device=b.xyz.device),dim=-1)
            raw=self.common.eval_sh(round(math.sqrt(b.sh.shape[-1])-1),b.sh,direction)+.5
        delta=self.dc_delta.detach() if mode=='occlusion' else self.dc_delta
        colors=raw.index_add(0,self.rgb_indices,delta*C0).clamp_min(0)
        return self.common.raster(camera,self.data(mode),colors)

    @torch.no_grad()
    def constrain(self):
        self.dc_delta.clamp_(-self.cfg['max_rgb_change']/C0,self.cfg['max_rgb_change']/C0)
        self.opacity_delta.copy_(torch.maximum(self.lower,torch.minimum(self.upper,self.opacity_delta)))


def load_views(common,folder):
    views=[]
    for record in read_json(folder/'manifest.json')['views']:
        cache=np.load(folder/'cache'/f"{record['name']}.npz")
        view=dict(record,camera_object=common.Camera.from_record(record['camera']))
        for key in ('object_alpha','visible_initial','background_initial','background_train_mask',
                    'visibility_weight','unknown_region','legal_foreground','old_projection','local_alpha'):
            view[key]=torch.tensor(cache[key],device='cuda',dtype=torch.float32)
        view['original_rgb']=torch.tensor(cache['original_rgb'].transpose(2,0,1).copy(),device='cuda',dtype=torch.float32)/255
        for key in ('environment_mask','old_mask','new_mask'):
            view[key]=torch.tensor(cache[key],device='cuda',dtype=torch.float32)
        view['preserve_mask']=view['environment_mask']*(view['local_alpha']>.005)
        views.append(view)
    return views


@torch.no_grad()
def evaluate_permissioned(model,obj,views,common):
    rows=[]
    for view in views:
        bg=model.render(view['camera_object'])
        visible=object_contribution(common,view['camera_object'],model.data(),obj)[0]
        mask=view['new_mask']; weight=view['visibility_weight']; env=view['environment_mask']
        change=(bg-view['background_initial']).abs().mean(0); legal=view['legal_foreground']
        rows.append({'name':view['name'],'split':view['split'],
            'visibility':float((visible*mask).sum()/(view['object_alpha']*mask).sum().clamp_min(1)),
            'trusted_visibility':float((visible*weight).sum()/(view['object_alpha']*weight).sum().clamp_min(1)),
            'environment_l1':float(masked_l1(bg,view['original_rgb'],view['background_train_mask'])),
            'outside_union_mae':float((change*env).sum()/env.sum().clamp_min(1)),
            'outside_union_changed_005':float(((change>.05)*env).sum()/env.sum().clamp_min(1)),
            'legal_foreground_pixels':int(legal.sum()),
            'legal_foreground_mae':float((change*legal).sum()/legal.sum().clamp_min(1))})
    summary={}
    for split in ('train','validation','anchor'):
        subset=[r for r in rows if r['split']==split]
        summary[split]={key:float(np.mean([r[key] for r in subset])) for key in rows[0] if key not in ('name','split')}
    return {'summary':summary,'views':rows}


def optimize_background(ctx):
    """Bounded dual-mask M4; only local background SH0 and opacity are trainable."""
    cfg=optimizer_config(ctx.cfg['m4'].get('local_background',{}))
    save_every=ctx.cfg['m4'].get('save_every',0)
    if type(save_every) is not int or save_every<0 or (save_every and save_every%cfg['evaluate_every']):
        raise ValueError('save_every must be zero or a positive multiple of evaluate_every')
    out=ctx.stage('m4')/'background_optimization'; out.mkdir(parents=True,exist_ok=True)
    folder=ensure_m4_supervision(ctx)
    manifest=read_json(folder/'manifest.json')
    background=ctx.stage('m1')/'background_gaussians.ply'; object_path=integrated_object(ctx)
    for key,path in {'background':background,'object':object_path,'mask':ctx.stage('m2')/'object_mask.png',
                     'camera':ctx.stage('m2')/'anchor_camera.json','manifest':ctx.stage('m1')/'manifest.json'}.items():
        if sha256(path)!=manifest['source_hashes'][key]: raise ValueError('Stale background-only input: '+key)
    signature=hashlib.sha256(json.dumps({'config':cfg,'visibility':manifest['signature']},sort_keys=True).encode()).hexdigest()
    if (out/'summary.json').exists():
        previous=read_json(out/'summary.json')
        compatible=(previous.get('method')==UNION_SCHEME and optimizer_config(previous['config'])==cfg
                    and previous.get('source_hashes')==manifest['source_hashes']
                    and previous['signature']==hashlib.sha256(json.dumps({'config':previous['config'],
                        'visibility':manifest['signature']},sort_keys=True).encode()).hexdigest())
        if previous['signature']!=signature and not compatible:
            raise ValueError('M4 cache mismatch; use a new experiment directory')
        assert previous['background_sha256']==sha256(out/'background_optimized.ply')
        return out/'background_optimized.ply'
    if (out/'training.jsonl').exists(): raise RuntimeError('Partial short run exists; use a new output directory instead of silently restarting')
    c=gsle(ctx); torch.manual_seed(ctx.cfg['project']['seed'])
    rng=np.random.default_rng(ctx.cfg['project']['seed'])
    rgb_ids=np.load(folder/'rgb_indices.npy'); occ_ids=np.load(folder/'occlusion_indices.npy')
    if len(occ_ids)==0: raise ValueError('No high-confidence multiview occluders; background remains unchanged')
    bg=c.GaussianData.load(background); obj=c.GaussianData.load(object_path)
    assert all(not getattr(obj,n).requires_grad for n in ('xyz','log_scale','rotation','opacity_logit','sh'))
    model=PermissionedBackground(bg,rgb_ids,occ_ids,c,cfg)
    views=load_views(c,folder); training=[v for v in views if v['split']=='train']; anchor=views[0]
    optimizer=torch.optim.Adam([{'params':[model.dc_delta],'lr':cfg['color_lr']},
                                {'params':[model.opacity_delta],'lr':cfg['opacity_lr']}])
    baseline=evaluate_permissioned(model,obj,views,c); write_json(out/'baseline.json',baseline)
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_step=0
    def objective(metrics):
        s=metrics['summary']; return s['validation']['environment_l1']+cfg['visibility_weight']*(1-s['validation']['trusted_visibility'])
    best_objective=objective(baseline); history=[]; start=time.time(); order=[]; start_step=0
    from .m4_resume import load_resume,restore_training_state
    resumed=load_resume(ctx,cfg,manifest)
    if resumed is not None:
        start_step=int(resumed['step']); best=resumed['best_state']; best_step=int(resumed['best_step'])
        if best_step:
            model.load_state_dict(best,strict=True)
            best_objective=objective(evaluate_permissioned(model,obj,views,c))
        order=restore_training_state(resumed,model,optimizer,rng,len(training))
        resume_metrics=evaluate_permissioned(model,obj,views,c)
        write_json(out/'resume_metrics.json',resume_metrics)
        write_json(out/'resume_verification.json',{'source':ctx.cfg['m4']['resume'],
            'start_step':start_step,'next_step':start_step+1,'target_step':cfg['iterations'],
            'model_state_bitwise_restored':True,'optimizer_state_bitwise_restored':True,
            'numpy_rng_restored':True,'torch_rng_restored':True,'cuda_rng_restored':True,
            'frozen_buffers_verified':True,'remaining_view_order':[int(i) for i in order],
            'optimizer_steps':[int(s['step'].item()) for s in optimizer.state.values()],
            'best_step':best_step,'new_training_steps':cfg['iterations']-start_step})
        print('Resumed actual M4 checkpoint at step',start_step,'target',cfg['iterations'],flush=True)
        del resumed
    for step in range(start_step+1,cfg['iterations']+1):
        if not order: order=list(rng.permutation(len(training)))
        view=training[order.pop()]; optimizer.zero_grad(set_to_none=True)
        photo=masked_l1(model.render(view['camera_object'],'rgb'),view['original_rgb'],view['background_train_mask'])
        alpha=(model.initial_logits+model.opacity_delta).sigmoid()
        rgb_reg=((alpha-model.initial_alpha).square()*model.rgb_permission).sum()/model.rgb_permission.sum().clamp_min(1)
        color_reg=(model.dc_delta*C0).square().mean()
        (cfg['rgb_weight']*photo+cfg['opacity_preserve_weight']*rgb_reg+cfg['color_preserve_weight']*color_reg).backward()
        visview=anchor if step%4==0 else view
        vis,_=visibility_loss(c,visview['camera_object'],model.data('occlusion'),obj,
                             visview['object_alpha'][None],visview['visibility_weight'])
        guard_image=model.render(visview['camera_object'],'occlusion')
        environment_guard=masked_l1(guard_image,visview['background_initial'],visview['preserve_mask'])
        legal_guard=masked_l1(guard_image,visview['background_initial'],visview['legal_foreground'])
        alpha=(model.initial_logits+model.opacity_delta).sigmoid()
        occ_reg=((alpha-model.initial_alpha).square()*model.occlusion_permission).sum()/model.occlusion_permission.sum().clamp_min(1)
        ramp=min(1.,step/max(1,cfg['warmup_steps']))
        (cfg['visibility_weight']*ramp*vis+cfg['occlusion_environment_guard_weight']*environment_guard+
         cfg['legal_foreground_guard_weight']*legal_guard+cfg['occlusion_regularization']*occ_reg).backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Invalid short-run gradient')
        optimizer.step(); model.constrain()
        item={'step':step,'rgb_view':view['name'],'visibility_view':visview['name'],
            'environment_l1':float(photo.detach()),'visibility_loss':float(vis.detach()),
            'environment_guard':float(environment_guard.detach()),'legal_guard':float(legal_guard.detach()),
            'elapsed_seconds':time.time()-start}
        with (out/'training.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
        del photo,rgb_reg,color_reg,vis,guard_image,environment_guard,legal_guard,occ_reg,alpha
        if step%25==0: print(item,flush=True)
        if step%cfg['evaluate_every']==0 or step==cfg['iterations']:
            metrics=evaluate_permissioned(model,obj,views,c)
            val=metrics['summary']['validation']; initial=baseline['summary']['validation']
            acceptable=val['environment_l1']<=initial['environment_l1']*(1+cfg['holdout_max_relative_degradation'])+.0002
            acceptable&=all(v['outside_union_mae']<=cfg['max_outside_union_mae'] and v['legal_foreground_mae']<=cfg['max_legal_foreground_mae'] for v in metrics['views'])
            score=objective(metrics); metrics.update(step=step,acceptable=acceptable,selection_objective=score)
            history.append(metrics); write_json(out/'optimization_log.json',{'config':cfg,'evaluations':history})
            torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'signature':signature},out/'checkpoint_latest.pth')
            if acceptable and score<best_objective:
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_step=step; best_objective=score
            print('short evaluation',step,'accepted',acceptable,metrics['summary'],flush=True)
        if save_every and (step%save_every==0 or step==cfg['iterations']):
            from .m4_snapshots import save_iteration_result
            save_iteration_result(ctx,model,obj,views,c,step,metrics,optimizer,rng,order,signature,best,best_step)
    model.load_state_dict(best); final=evaluate_permissioned(model,obj,views,c)
    vertices=read_vertices(background); updated=vertices.copy(); ids=model.indices.cpu().numpy()
    updated['opacity'][ids]=(model.initial_logits+model.opacity_delta).detach().cpu().numpy()[:,0]
    delta=model.dc_delta.detach().cpu().numpy()
    for k in range(3): updated[f'f_dc_{k}'][rgb_ids]+=delta[:,k]
    save_vertices(out/'background_optimized.ply',updated)
    check=read_vertices(out/'background_optimized.ply'); frozen=np.ones(len(vertices),bool); frozen[ids]=False
    assert vertices[frozen].tobytes()==check[frozen].tobytes()
    for field in vertices.dtype.names:
        if field not in ('opacity','f_dc_0','f_dc_1','f_dc_2'): assert vertices[field].tobytes()==check[field].tobytes()
    not_rgb=np.ones(len(vertices),bool); not_rgb[rgb_ids]=False
    for k in range(3): assert vertices[f'f_dc_{k}'][not_rgb].tobytes()==check[f'f_dc_{k}'][not_rgb].tobytes()
    compose(out/'background_optimized.ply',object_path,out/'scene_visibility_optimized.ply',out/'composition.json')
    assert sha256(background)==manifest['source_hashes']['background'] and sha256(object_path)==manifest['source_hashes']['object']
    np.save(out/'optimized_indices.npy',ids); np.save(out/'rgb_indices.npy',rgb_ids); np.save(out/'occlusion_indices.npy',occ_ids)
    torch.save({'state':best,'step':best_step,'signature':signature},out/'checkpoint_best.pth')
    summary={'method':UNION_SCHEME,'background_only':True,'signature':signature,'config':cfg,
        'iterations':cfg['iterations'],'start_step':start_step,'executed_steps':cfg['iterations']-start_step,
        'resume':ctx.cfg['m4'].get('resume'),'selected_step':best_step,'seconds':time.time()-start,
        'rgb_points':len(rgb_ids),'occlusion_points':len(occ_ids),'occlusion_without_rgb_points':int((~np.isin(occ_ids,rgb_ids)).sum()),
        'background_points':len(vertices),'optimized_points':len(ids),'deleted_points':0,'added_points':0,
        'object_all_parameters_bitwise_preserved':True,'background_geometry_bitwise_preserved':True,
        'unselected_background_bitwise_preserved':True,'non_rgb_colors_bitwise_preserved':True,
        'source_hashes':manifest['source_hashes'],'object_sha256':sha256(object_path),
        'background_sha256':sha256(out/'background_optimized.ply'),'baseline':baseline,'final':final,
        'visibility_folder':str(folder),'di_sds_executed':False,'qwen_calls':0,'trellis_calls':0,
        'selection':'Best guarded checkpoint within configured short-run budget; baseline is fallback if none improves validation.'}
    write_json(out/'summary.json',summary); write_json(out/'final_metrics.json',final)
    del bg,obj,model,optimizer,views,best,vertices,updated,check; torch.cuda.empty_cache()
    return out/'background_optimized.ply'
