"""Stage A: photo-supervised local SH0/opacity optimization, fixed topology."""
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
from .multiview_masks import prepare_multiview
from .visibility_loss import masked_l1, visibility_loss, object_contribution

C0=0.28209479177387814
DEFAULTS={'enabled':True,'iterations':1000,'color_lr':.004,'opacity_lr':.015,
    'rgb_weight':1.,'visibility_weight':.3,'outside_preserve_weight':.2,
    'opacity_preserve_weight':.05,'color_preserve_weight':.02,
    'max_rgb_change':.18,'max_opacity_change':.12,'candidate_max_opacity_change':.6,
    'warmup_steps':100,'evaluate_every':100,'holdout_max_relative_degradation':.01}


class LocalBackground(torch.nn.Module):
    """Deltas are zero at initialization; untouched Gaussian rows never round-trip."""
    def __init__(self,original,indices,candidates,common,cfg):
        super().__init__(); self.original=original; self.common=common; self.cfg=cfg
        device=original.xyz.device
        self.register_buffer('indices',torch.as_tensor(indices,device=device,dtype=torch.long))
        self.register_buffer('initial_logits',original.opacity_logit[self.indices].detach().clone())
        self.register_buffer('initial_alpha',self.initial_logits.sigmoid())
        self.dc_delta=torch.nn.Parameter(torch.zeros((len(indices),3),device=device))
        self.opacity_delta=torch.nn.Parameter(torch.zeros_like(self.initial_logits))
        limit=torch.tensor(np.where(np.isin(indices,candidates),cfg['candidate_max_opacity_change'],cfg['max_opacity_change']),device=device,dtype=self.initial_alpha.dtype)[:,None]
        low=torch.logit((self.initial_alpha-limit).clamp(1e-6,1-1e-6))-self.initial_logits
        high=torch.logit((self.initial_alpha+limit).clamp(1e-6,1-1e-6))-self.initial_logits
        self.register_buffer('lower',torch.minimum(low,torch.zeros_like(low)))
        self.register_buffer('upper',torch.maximum(high,torch.zeros_like(high)))

    def data(self):
        b=self.original
        opacity=b.opacity_logit.index_copy(0,self.indices,self.initial_logits+self.opacity_delta)
        return self.common.GaussianData(b.xyz,b.log_scale,b.rotation,opacity,b.sh)

    def render(self,camera):
        b=self.original
        with torch.no_grad():
            direction=F.normalize(b.xyz-torch.as_tensor(camera.center,device=b.xyz.device),dim=-1)
            raw=self.common.eval_sh(round(math.sqrt(b.sh.shape[-1])-1),b.sh,direction)+.5
        colors=raw.index_add(0,self.indices,self.dc_delta*C0).clamp_min(0)
        return self.common.raster(camera,self.data(),colors)

    def regularization(self):
        alpha=(self.initial_logits+self.opacity_delta).sigmoid()
        return (alpha-self.initial_alpha).square().mean(),(self.dc_delta*C0).square().mean()

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
        # Preserve image content outside old edit projection and new silhouette.
        view['preserve_mask']=(1-view['old_projection'])*(view['object_alpha']<.05)*(view['local_alpha']>.005)
        views.append(view)
    return views


@torch.no_grad()
def evaluate(model,obj,views,common):
    rows=[]
    for view in views:
        camera=view['camera_object']; bg=model.render(camera)
        visible=object_contribution(common,camera,model.data(),obj)[0]
        mask=view['object_alpha']>.25; weight=view['visibility_weight']
        denom=(view['object_alpha']*mask).sum().clamp_min(1)
        outside=~mask; change=(bg-view['background_initial']).abs().mean(0)
        row={'name':view['name'],'split':view['split'],
            'visibility':float((visible*mask).sum()/denom),
            'trusted_visibility':float((visible*weight).sum()/(view['object_alpha']*weight).sum().clamp_min(1)),
            'background_l1':float(masked_l1(bg,view['original_rgb'],view['background_train_mask'])),
            'outside_object_mae':float((change*outside).sum()/outside.sum().clamp_min(1)),
            'outside_changed_005_fraction':float(((change>.05)&outside).sum()/outside.sum().clamp_min(1)),
            'legal_foreground_change':float((change*view['legal_foreground']).sum()/view['legal_foreground'].sum().clamp_min(1))}
        rows.append(row)
    summary={}
    for split in ('train','validation','anchor'):
        subset=[r for r in rows if r['split']==split]
        summary[split]={key:float(np.mean([r[key] for r in subset])) for key in rows[0] if key not in ('name','split')}
    return {'summary':summary,'views':rows}


def optimize_background(ctx):
    cfg={**DEFAULTS,**ctx.cfg['m4'].get('local_background',{})}
    if cfg['iterations']<1 or cfg['evaluate_every']<1:
        raise ValueError('Stage A iteration/evaluation counts must be positive')
    if not 0<cfg['max_opacity_change']<=cfg['candidate_max_opacity_change']<1:
        raise ValueError('Opacity bounds must preserve a non-deletion optimization policy')
    out=ctx.stage('m4')/'background_optimization'; out.mkdir(parents=True,exist_ok=True)
    # Fallback analysis is placed in M4, never in a reused/symlinked old M3.
    folder=ctx.stage('m3')/'multiview_visibility'
    if not (folder/'manifest.json').exists(): folder=prepare_multiview(ctx,out/'visibility_prepare')
    manifest=read_json(folder/'manifest.json')
    background=ctx.stage('m1')/'background_gaussians.ply'; object_path=ctx.stage('m3')/'object_world.ply'
    expected={'background':background,'object':object_path,'mask':ctx.stage('m2')/'object_mask.png',
              'camera':ctx.stage('m2')/'anchor_camera.json','manifest':ctx.stage('m1')/'manifest.json'}
    for key,path in expected.items():
        if sha256(path)!=manifest['source_hashes'][key]: raise ValueError('Stale multiview input: '+key)
    signature=hashlib.sha256(json.dumps({'config':cfg,'visibility_signature':manifest['signature'],
        'background':sha256(background),'object':sha256(object_path)},sort_keys=True).encode()).hexdigest()
    if (out/'summary.json').exists():
        summary=read_json(out/'summary.json'); assert summary['signature']==signature,'Stage A input/config mismatch'
        assert sha256(out/'background_optimized.ply')==summary['background_sha256']
        return out/'background_optimized.ply'
    c=gsle(ctx); torch.manual_seed(ctx.cfg['project']['seed']); rng=np.random.default_rng(ctx.cfg['project']['seed'])
    ids=np.load(folder/'local_indices.npy'); candidates=np.load(folder/'occlusion_candidate_indices.npy')
    if len(ids)==0: raise ValueError('No background Gaussians have reliable multi-view observation support')
    bg=c.GaussianData.load(background); obj=c.GaussianData.load(object_path)
    model=LocalBackground(bg,ids,candidates,c,cfg); views=load_views(c,folder)
    training=[v for v in views if v['split']=='train']; anchor=views[0]
    optimizer=torch.optim.Adam([{'params':[model.dc_delta],'lr':cfg['color_lr']},
                               {'params':[model.opacity_delta],'lr':cfg['opacity_lr']}])
    baseline=evaluate(model,obj,views,c); write_json(out/'baseline.json',baseline)
    best_objective=float('inf'); best_step=0; best_state=None; history=[]; start=time.time(); first=0; previous_elapsed=0
    latest=out/'checkpoint_latest.pth'
    if latest.exists():
        saved=torch.load(latest,map_location=model.indices.device,weights_only=False)
        assert saved['signature']==signature,'Stage A checkpoint input/config mismatch'
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer']); first=saved['step']
        if 'rng' not in saved: raise ValueError('Legacy incomplete Stage A checkpoint lacks RNG; use an independent output directory')
        rng.bit_generator.state=saved['rng']; previous_elapsed=saved.get('elapsed_seconds',0)
        history=read_json(out/'optimization_log.json')['evaluations']
        best_path=out/'checkpoint_best.pth'
        if best_path.exists():
            best=torch.load(best_path,map_location='cpu',weights_only=False)
            assert best['signature']==signature
            best_state=best['state']; best_step=best['step']
            best_objective=next(row['selection_objective'] for row in history if row['step']==best_step)
    for step in range(first+1,cfg['iterations']+1):
        view=training[int(rng.integers(len(training)))]; camera=view['camera_object']
        optimizer.zero_grad(set_to_none=True); rendered=model.render(camera)
        photo=masked_l1(rendered,view['original_rgb'],view['background_train_mask'])
        preservation=masked_l1(rendered,view['background_initial'],view['preserve_mask'])
        opacity_reg,color_reg=model.regularization()
        loss=cfg['rgb_weight']*photo+cfg['outside_preserve_weight']*preservation+cfg['opacity_preserve_weight']*opacity_reg+cfg['color_preserve_weight']*color_reg
        loss.backward(); del loss,rendered
        visview=anchor if step%4==0 else view
        vis,_=visibility_loss(c,visview['camera_object'],model.data(),obj,visview['object_alpha'][None],visview['visibility_weight'])
        ramp=min(1.,max(0.,(step-cfg['warmup_steps'])/max(1,cfg['warmup_steps'])))
        (vis*cfg['visibility_weight']*ramp).backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Nonfinite background gradient')
        optimizer.step(); model.constrain()
        record={'step':step,'background_l1':float(photo),'visibility_loss':float(vis),
            'preserve_image_l1':float(preservation),'opacity_regularization':float(opacity_reg),
            'color_regularization':float(color_reg),'elapsed_seconds':previous_elapsed+time.time()-start}
        with (out/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if step%25==0: print(record,flush=True)
        if step%cfg['evaluate_every']==0 or step==cfg['iterations']:
            metrics=evaluate(model,obj,views,c); metrics.update(step=step,elapsed_seconds=time.time()-start)
            val=metrics['summary']['validation']; initial=baseline['summary']['validation']
            acceptable=val['background_l1']<=initial['background_l1']*(1+cfg['holdout_max_relative_degradation'])
            # Selection uses only held-out real-photo error and visibility, not screenshots.
            objective=val['background_l1']+cfg['visibility_weight']*(1-val['trusted_visibility'])
            metrics.update(acceptable=acceptable,selection_objective=objective); history.append(metrics)
            write_json(out/'optimization_log.json',{'config':cfg,'baseline':baseline,'evaluations':history})
            if acceptable and objective<best_objective:
                best_objective=objective; best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                temporary=out/'checkpoint_best.tmp'
                torch.save({'state':best_state,'step':step,'signature':signature},temporary); temporary.replace(out/'checkpoint_best.pth')
            temporary=out/'checkpoint_latest.tmp'
            torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'signature':signature,
                'rng':rng.bit_generator.state,'elapsed_seconds':previous_elapsed+time.time()-start},temporary)
            temporary.replace(latest)
            print('evaluation',step,metrics['summary'],'accepted',acceptable,flush=True)
    if best_state is None: raise RuntimeError('No checkpoint passed held-out background preservation; inspect diagnostics')
    model.load_state_dict(best_state); final=evaluate(model,obj,views,c)
    vertices=read_vertices(background); updated=vertices.copy()
    updated['opacity'][ids]=(model.initial_logits+model.opacity_delta).detach().cpu().numpy()[:,0]
    delta=model.dc_delta.detach().cpu().numpy()
    for channel in range(3): updated[f'f_dc_{channel}'][ids]+=delta[:,channel]
    save_vertices(out/'background_optimized.ply',updated)
    check=read_vertices(out/'background_optimized.ply'); frozen=np.ones(len(vertices),bool); frozen[ids]=False
    assert check[frozen].tobytes()==vertices[frozen].tobytes()
    for field in vertices.dtype.names:
        if field not in ('opacity','f_dc_0','f_dc_1','f_dc_2'): assert check[field].tobytes()==vertices[field].tobytes()
    np.save(out/'optimized_indices.npy',ids)
    compose(out/'background_optimized.ply',object_path,out/'scene_visibility_optimized.ply',out/'composition.json')
    alpha0=model.initial_alpha[:,0].cpu().numpy(); alpha1=(model.initial_logits+model.opacity_delta).sigmoid().detach().cpu().numpy()[:,0]
    summary={'signature':signature,'config':cfg,'selected_step':best_step,'iterations':cfg['iterations'],
        'seconds':previous_elapsed+time.time()-start,'background_points':len(vertices),'optimized_points':len(ids),
        'unobserved_local_frozen':manifest['local_unsupported_count'],'deleted_points':0,'added_points':0,
        'background_geometry_bitwise_preserved':True,'outside_local_bitwise_preserved':True,
        'original_background_sha256':sha256(background),'object_sha256':sha256(object_path),
        'background_sha256':sha256(out/'background_optimized.ply'),'baseline':baseline['summary'],'final':final['summary'],
        'opacity_change':{'mean_absolute_local':float(abs(alpha1-alpha0).mean()),'mean_local':float((alpha1-alpha0).mean()),
            'local_drop_over_02':int((alpha0-alpha1>.2).sum()),'local_new_below_01':int(((alpha1<.1)&(alpha0>=.1)).sum())},
        'unknown_policy':manifest['unknown_policy'],'visibility_folder':str(folder)}
    write_json(out/'final_metrics.json',final); write_json(out/'summary.json',summary)
    del bg,obj,model,optimizer,views,best_state; torch.cuda.empty_cache()
    return out/'background_optimized.ply'
