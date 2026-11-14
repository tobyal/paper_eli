"""Case00 M4 figure: read existing 2000-step candidates, render and audit only."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=3);a=ap.parse_args()
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),SWAP_DISABLE_QWEN='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    import numpy as np
    import torch
    from PIL import Image
    from scipy.special import expit
    from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
    from industrial_swap_v2.core.scene import read_vertices

    base=root/'industrial_swap_v2/outputs/registration_all_cases_v1/case00'
    old=root/'industrial_swap_v2/outputs/paper_figures_v4/F14_background_refinement'
    out=old.parent/'F14_occlusion_refinement_revised'
    if out.exists():raise FileExistsError('Preserve previous output: '+str(out))
    out.mkdir();panels=out/'panels';panels.mkdir();records=out/'records';records.mkdir();(out/'code').mkdir()
    full=base/'m4';rgb=old/'rgb_only_run/m4'
    supers=full/'multiview_supervision';manifest=read_json(supers/'manifest.json')
    summaries={};logs={};baselines={}
    for name,folder in [('RGB-only',rgb),('Full M4 Candidate',full)]:
        train=folder/'background_optimization'
        summaries[name]=read_json(train/'summary.json')
        logs[name]=read_json(train/'optimization_log.json')['evaluations']
        baselines[name]=read_json(train/'baseline.json')
        assert summaries[name]['iterations']==2000 and logs[name][-1]['step']==2000
        for filename in ('summary.json','optimization_log.json','baseline.json'):
            shutil.copy2(train/filename,records/(('rgb_' if folder==rgb else 'full_')+filename))
    assert summaries['RGB-only']['source_hashes']==summaries['Full M4 Candidate']['source_hashes']
    assert baselines['RGB-only']==baselines['Full M4 Candidate']
    backgrounds={'A':base/'m1/background_gaussians.ply',
        'B':rgb/'snapshots/iteration_002000/background_optimized.ply',
        'C':full/'background_optimization/background_candidate_step2000.ply'}
    object_path=full/'object_world.ply'
    ids=np.load(supers/'occlusion_indices.npy');rgb_ids=np.load(supers/'rgb_indices.npy')
    source_paths=[*backgrounds.values(),object_path,supers/'occlusion_indices.npy',supers/'rgb_indices.npy',
        supers/'manifest.json',full/'summary.json',old/'rgb_only_run/ablation.json',
        old/'figure_F14.png',old/'figure_F14.pdf']
    for folder in (rgb,full):
        source_paths.extend(folder/'background_optimization'/f for f in ('summary.json','baseline.json','optimization_log.json'))
    hashes={str(p):sha256(p) for p in source_paths}
    assert sha256(object_path)==summaries['Full M4 Candidate']['object_sha256']
    assert sha256(backgrounds['A'])==summaries['Full M4 Candidate']['source_hashes']['background']
    np.save(records/'occlusion_indices.npy',ids)
    raw0=read_vertices(backgrounds['A']);alpha0=expit(np.asarray(raw0['opacity'][ids],np.float64))
    opacity={'A':alpha0};geometry_checks={}
    geo=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    for key in ('B','C'):
        raw=read_vertices(backgrounds[key]);assert len(raw)==len(raw0)
        geometry_checks[key]={k:raw[k].tobytes()==raw0[k].tobytes() for k in geo}
        assert all(geometry_checks[key].values())
        opacity[key]=expit(np.asarray(raw['opacity'][ids],np.float64));del raw
    assert np.max(opacity['C']-opacity['A'])<1e-6
    table=[]
    for j,index in enumerate(ids):
        table.append({'background_index':int(index),'x':float(raw0['x'][index]),'y':float(raw0['y'][index]),'z':float(raw0['z'][index]),
            'alpha_initial':float(opacity['A'][j]),'alpha_rgb_only':float(opacity['B'][j]),'alpha_full_candidate':float(opacity['C'][j]),
            'delta_alpha_rgb':float(opacity['B'][j]-opacity['A'][j]),'delta_alpha_full':float(opacity['C'][j]-opacity['A'][j]),
            'also_in_rgb_set':bool(index in rgb_ids)})
    def csvout(path,rows):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csvout(out/'occlusion_gaussians.csv',table)
    background_count=len(raw0);del raw0
    # Extract raw held-out evaluations; no fitting, resampling or smoothing.
    visibility_rows=[];environment_rows=[];guards={}
    for method in summaries:
        summary=summaries[method];cfg=summary['config'];baseline=baselines[method]
        checkpoints=[{'step':0,**baseline,'acceptable':True}]+logs[method]
        valid_names=[v['name'] for v in baseline['views'] if v['split']=='validation']
        for e in checkpoints:
            v=e['summary']['validation']
            visibility_rows.append({'method':method,'step':e['step'],'split':'validation','n_views':len(valid_names),
                'trusted_visibility':v['trusted_visibility'],'object_visibility':v['visibility'],'guard_acceptable':e['acceptable']})
            environment_rows.append({'method':method,'step':e['step'],'split':'validation','n_views':len(valid_names),
                'heldout_environment_rgb_l1':v['environment_l1'],'outside_union_rgb_mae':v['outside_union_mae'],
                'legal_foreground_mae':v['legal_foreground_mae'],'guard_acceptable':e['acceptable']})
        last=logs[method][-1];limit=baseline['summary']['validation']['environment_l1']*(1+cfg['holdout_max_relative_degradation'])+.0002
        violations=[{'name':v['name'],'split':v['split'],'outside_union_mae':v['outside_union_mae'],
                     'legal_foreground_mae':v['legal_foreground_mae'],
                     'outside_failed':v['outside_union_mae']>cfg['max_outside_union_mae'],
                     'legal_failed':v['legal_foreground_mae']>cfg['max_legal_foreground_mae']}
                    for v in last['views'] if v['outside_union_mae']>cfg['max_outside_union_mae'] or v['legal_foreground_mae']>cfg['max_legal_foreground_mae']]
        guards[method]={'candidate_step':2000,'candidate_accepted':last['acceptable'],
            'guarded_selected_step':summary['selected_step'],
            'accepted_nonzero_steps':[e['step'] for e in logs[method] if e['acceptable']],
            'heldout_rgb_limit':limit,'heldout_rgb_final':last['summary']['validation']['environment_l1'],
            'heldout_rgb_passed':last['summary']['validation']['environment_l1']<=limit,
            'per_view_outside_union_limit':cfg['max_outside_union_mae'],
            'per_view_legal_foreground_limit':cfg['max_legal_foreground_mae'],
            'violating_views':violations,'guarded_background':str((rgb if method=='RGB-only' else full)/'background_optimization/background_optimized.ply')}
        # Verify the selected fallback is actually the unmodified background.
        if summary['selected_step']==0:
            guards[method]['guarded_background_matches_step0']=sha256(guards[method]['guarded_background'])==sha256(backgrounds['A'])
            assert guards[method]['guarded_background_matches_step0']
    csvout(out/'visibility_metrics.csv',visibility_rows);csvout(out/'environment_metrics.csv',environment_rows)
    write_json(out/'guard_status.json',guards)
    print('LOGS',guards.keys(),'occlusion gaussians',len(ids),flush=True)

    ctx=Context(base/'config.yaml');ctx.cfg['project']['gpu']=a.gpu;c=gsle(ctx)
    torch.cuda.set_per_process_memory_fraction(.45)
    view=manifest['views'][0];cache=np.load(supers/'cache'/f"{view['name']}.npz")
    assert np.array_equal(cache['environment_mask'],~(cache['old_mask']|cache['new_mask']))
    cam_native=c.Camera.from_record(view['camera']);anchor=cam_native.resize(1024,1024)
    obj=c.GaussianData.load(object_path);up=torch.tensor(read_json(base/'m1/manifest.json')['world_up'],device='cuda')
    height=obj.xyz@up;base_flags=(height<torch.quantile(height,.2)).float()
    with torch.no_grad():
        packed=torch.stack((torch.ones_like(base_flags),base_flags,torch.zeros_like(base_flags)),1)
        alone=c.raster(anchor,obj,packed)
        alone_native=c.raster(cam_native,obj,packed)
    def bbox(mask,margin):
        yy,xx=np.where(mask)
        return [max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(1024,int(xx.max())+margin+1),min(1024,int(yy.max())+margin+1)]
    object_box=bbox(alone[0].cpu().numpy()>.25,0)
    mid=(np.array(object_box[:2])+object_box[2:])/2
    side=int(max(object_box[2]-object_box[0],object_box[3]-object_box[1])*1.38)
    main_box=[int(mid[0]-side/2),int(mid[1]-side/2),int(mid[0]+side/2),int(mid[1]+side/2)]
    main_box=[max(0,main_box[0]),max(0,main_box[1]),min(1024,main_box[2]),min(1024,main_box[3])]
    base_box=bbox(alone[1].cpu().numpy()>.10,25)
    def crop_camera(box,width):
        x0,y0,x1,y1=box;scale=width/(x1-x0);record=anchor.record()
        record.update(width=width,height=round((y1-y0)*scale),fx=anchor.fx*scale,fy=anchor.fy*scale,
                      cx=(anchor.cx-x0)*scale,cy=(anchor.cy-y0)*scale)
        return c.Camera.from_record(record)
    maincam=crop_camera(main_box,1100);detailcam=crop_camera(base_box,1000)
    # Mask panels show the exact cached anchor definitions, not hand redrawing.
    for key in ('old_mask','new_mask','environment_mask'):
        image=Image.fromarray(cache[key].astype(np.uint8)*255)
        image.save(panels/(key+'_full.png'))
        image=image.resize((1024,1024),Image.Resampling.NEAREST).crop(main_box)
        image.save(panels/(key+'.png'))
    local_metrics=[];initial_detail=None;adjustment={}
    with torch.no_grad():
        for key,path in backgrounds.items():
            bg=c.GaussianData.load(path);joint=c.concatenate(bg,obj)
            main_rgb=c.render_joint(maincam,bg,obj,aux=False)['rgb']
            detail_rgb=c.render_joint(detailcam,bg,obj,aux=False)['rgb']
            c.save_image(panels/f'{key}_scene.png',main_rgb);c.save_image(panels/f'{key}_base.png',detail_rgb)
            colors=torch.zeros((len(joint.xyz),3),device='cuda')
            colors[len(bg.xyz):,0]=1;colors[len(bg.xyz):,1]=base_flags
            colors[torch.as_tensor(ids,device='cuda'),2]=1
            contrib=c.raster(cam_native,joint,colors)
            w=torch.tensor(cache['visibility_weight'],device='cuda')
            record={'method':key,'base_visibility':float(contrib[1].sum()/alone_native[1].sum().clamp_min(1)),
                    'trusted_object_visibility_anchor':float((contrib[0]*w).sum()/(alone_native[0]*w).sum().clamp_min(1))}
            local_metrics.append(record)
            if key=='A':
                initial_detail=detail_rgb.cpu().numpy().transpose(1,2,0)
                delta=np.maximum(opacity['A']-opacity['C'],0)
                colors.zero_();ii=torch.as_tensor(ids,device='cuda')
                colors[ii,0]=torch.as_tensor(delta,device='cuda',dtype=torch.float32)
                colors[ii,1]=1
                proj=c.raster(detailcam,joint,colors).cpu().numpy()
                mass=proj[1];mean=np.divide(proj[0],mass,out=np.zeros_like(mass),where=mass>1e-8)
                adjustment={'initial_candidate_contribution':mass,'weighted_opacity_reduction':proj[0],
                            'mean_opacity_reduction':mean}
                overlay=initial_detail*(1-.85*mass[:,:,None])+np.array([1.,.10,.06])[None,None,:]*(.85*mass[:,:,None])
                Image.fromarray(np.round(overlay.clip(0,1)*255).astype(np.uint8)).save(panels/'occluders_projection.png')
                # Same real maximum-contribution location marked across A/B/C details.
                base_alone=c.raster(detailcam,obj,packed)[1].cpu().numpy()
                observation=mass*base_alone
                yy,xx=np.unravel_index(np.argmax(observation),observation.shape)
                adjustment['base_alpha']=base_alone
                arrow_pixel=[int(xx),int(yy)]
            del bg,joint,main_rgb,detail_rgb,colors,contrib;gc.collect();torch.cuda.empty_cache()
            print('RENDER',key,record,flush=True)
    np.savez_compressed(out/'occlusion_projection.npz',**adjustment)
    csvout(out/'anchor_base_visibility.csv',local_metrics)
    copy_files=[('core/multiview_masks.py','multiview_masks.py'),('core/local_background_optimizer.py','local_background_optimizer.py'),
                ('core/visibility_loss.py','visibility_loss.py'),('core/m4_outputs.py','m4_outputs.py')]
    for rel,name in copy_files:shutil.copy2(root/'industrial_swap_v2'/rel,out/'code'/name)
    shutil.copy2(__file__,out/'code'/Path(__file__).name)
    changes={k:{'mean_alpha':float(v.mean()),'median_alpha':float(np.median(v)),
                'mean_alpha_reduction':float((alpha0-v).mean()),'n_reduced_over_1e6':int((alpha0-v>1e-6).sum()),
                'max_alpha_reduction':float((alpha0-v).max())} for k,v in opacity.items()}
    original_checks=read_json(full/'summary.json')['checks']
    preserved={p:sha256(p)==h for p,h in hashes.items()};assert all(preserved.values())
    sources={'case':'case00','full_run':str(base),'rgb_run':str(old/'rgb_only_run'),
        'input_sha256':hashes,'source_preservation':preserved,'background_paths':{k:str(p) for k,p in backgrounds.items()},
        'fixed_object':str(object_path),'fixed_object_sha256':sha256(object_path),'background_count':background_count,
        'object_count':len(obj.xyz),'occlusion_count':len(ids),'opacity_statistics':changes,
        'background_geometry_bitwise_equal':geometry_checks,'original_m4_export_checks':original_checks,
        'm4_ablation':read_json(old/'rgb_only_run/ablation.json'),
        'anchor_camera':anchor.record(),'native_evaluation_camera':cam_native.record(),
        'main_camera':maincam.record(),'detail_camera':detailcam.record(),
        'main_crop_xyxy_at1024':main_box,'detail_crop_xyxy_at1024':base_box,'observation_arrow_pixel_detail':arrow_pixel,
        'camera_note':'All methods use the same saved Anchor extrinsics and common crop intrinsics; close-ups are directly rasterized at high resolution, not repainted or upscaled screenshots.',
        'base_definition':'Lowest 20% of fixed object Gaussian centers along saved M1 world_up, exactly matching the original export definition.',
        'local_visibility':local_metrics,'curve_split':'validation','curve_view_count':10,'curve_sample_interval':50,
        'trusted_visibility_definition':'For each held-out view, sum(object contribution * trusted weight) / sum(standalone object alpha * trusted weight); then arithmetic mean over the same 10 views.',
        'environment_metric_definition':'Recorded held-out background RGB L1 against registered real photographs, weighted by background_train_mask (reliable, edge-protected local subset of the old/new union complement).',
        'mask_view':view,'mask_note':'Panel d displays exact cached Anchor mask definitions. Anchor has no real-photo RGB loss; environment training and held-out evaluation use registered photographic views.',
        'opacity_projection_definition':'Using original A compositing weights, mean alpha reduction = raster(candidate*(alpha0-alphaC))/raster(candidate). Candidate flag and actual baseline opacity/covariance define support. No geometric modification or deletion.',
        'no_new_training':True,'no_gaussian_geometry_edits':True,'no_added_or_deleted_gaussians':True,
        'guard_summary':guards,'script_sha256':sha256(__file__)}
    write_json(out/'sources.json',sources)
    print('F14 EXTRACTION COMPLETE',out,flush=True)


if __name__=='__main__':main()
