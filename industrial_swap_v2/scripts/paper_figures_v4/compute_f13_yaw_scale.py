"""Read-only loss evaluation around the accepted Case02 Gaussian pose."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=2);args=ap.parse_args()
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),SWAP_DISABLE_QWEN='1',
                      OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    import numpy as np
    import torch
    from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
    from industrial_swap_v2.core.m2_space import registration_space
    from industrial_swap_v2.core import gaussian_registration as reg

    source=root/'industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration'
    accepted=source.parent/'F13_case02_registration_final'
    out=source.parent/'F13_case02_registration_with_heatmap'
    if out.exists():raise FileExistsError('Keep existing results: '+str(out))
    out.mkdir();(out/'code').mkdir();(out/'records').mkdir();(out/'panels').mkdir()
    ctx=Context(source/'C/config.yaml');ctx.cfg['project']['gpu']=args.gpu
    c=gsle(ctx);torch.cuda.set_per_process_memory_fraction(.45)
    torch.manual_seed(42);np.random.seed(42)
    transform_path=source/'C/m4/transform.json';search_path=source/'C/m4/alignment/search.json'
    transform=read_json(transform_path);search=read_json(search_path);cfg=search['settings']
    assert cfg==transform['registration_settings']
    p0=np.asarray(transform['parameters_radians_logscale_dx_dz'],np.float64)
    assert np.allclose(p0,search['selected']['parameters'],atol=1e-12,rtol=0)
    asset=ctx.stage('m3')/'object_refined_local.ply'
    paths=[asset,ctx.stage('m2')/'anchor_camera.json',ctx.stage('m2')/'object_mask.png',
           ctx.stage('m2')/'object_rgba.png',ctx.stage('m2')/'generation_region.json',
           ctx.stage('m1')/'edit_region.json',ctx.stage('m1')/'manifest.json',
           source/'C/config.yaml',transform_path,search_path,source/'registration_metrics.csv',
           accepted/'figure_F13_final.png',accepted/'figure_F13_final.pdf',
           root/'experiment figure/F13/F13.png',root/'experiment figure/F13/F13.pdf',
           root/'industrial_swap_v2/core/gaussian_registration.py',root/'gsle/common.py']
    hashes={str(p):sha256(p) for p in paths}
    original_local=c.GaussianData.load(asset)
    space=registration_space(ctx);model=reg.Pose(c,original_local,space,cfg)
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    ref=reg.reference(ctx,camera,cfg['refine_resolution'],original_local.xyz.device)
    selected=model.tensor(p0)
    yaw0=float(transform['yaw_degrees']);scale0=float(transform['scale'])
    # Exact independent rasterizations, including the saved selected point.
    yaw_offsets=np.linspace(-45,45,61)
    ratios=np.round(np.arange(.70,1.061,.01),8)
    assert ratios[-1]*scale0 <= model.base_scale*cfg['max_scale_factor']
    assert ratios[0]*scale0 >= model.base_scale*cfg['min_scale_factor']
    yaws=yaw0+yaw_offsets;rows=[];start=time.monotonic()
    with torch.no_grad():
        _,m,_=reg.evaluate(model,selected,ref);selected_metrics=reg.numbers(m)
        assert abs(selected_metrics['objective']-search['selected']['objective'])<5e-5
        for si,ratio in enumerate(ratios):
            for yi,offset in enumerate(yaw_offsets):
                p=selected.clone();p[0]=selected[0]+math.radians(float(offset))
                p[1]=selected[1]+math.log(float(ratio))
                assert torch.equal(p[2:],selected[2:])
                _,metrics,_=reg.evaluate(model,p,ref);numbers=reg.numbers(metrics)
                assert np.isfinite(list(numbers.values())).all()
                rows.append({'scale_index':si,'yaw_index':yi,'yaw_degrees':float(yaws[yi]),
                    'scale_over_selected':float(ratio),'actual_scale':float(p[1].exp()),
                    'yaw_parameter_radians':float(p[0]),'logscale_parameter':float(p[1]),
                    'fixed_dx_fraction':float(p[2]),'fixed_dz_fraction':float(p[3]),**numbers})
            if si%6==0 or si==len(ratios)-1:
                print('GRID',si+1,'/',len(ratios),'points',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    with (out/'yaw_scale_objective.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fields=['objective','soft_iou','mask_iou','appearance_loss','chromaticity_loss',
            'foreground_rgb_mae','boundary_loss','center_penalty','volume_penalty']
    arrays={k:np.array([r[k] for r in rows]).reshape(len(ratios),len(yaws)) for k in fields}
    np.savez_compressed(out/'yaw_scale_grid.npz',yaw_degrees=yaws,scale_over_selected=ratios,**arrays)
    best=min(rows,key=lambda x:x['objective'])
    selected_row=next(r for r in rows if r['scale_over_selected']==1.0 and abs(r['yaw_degrees']-yaw0)<1e-10)
    assert abs(selected_row['objective']-selected_metrics['objective'])<1e-6
    preserved={p:sha256(p)==h for p,h in hashes.items()};assert all(preserved.values())
    summary={'case':'case02','source_experiment':str(source),'accepted_layout':str(accepted),
        'source_sha256':hashes,'source_files_unchanged':preserved,
        'transform_path':str(transform_path),'search_path':str(search_path),
        'transform_location_note':'Actual transform is C/m4/transform.json; it is not inside alignment/.',
        'asset':str(asset.resolve()),'gaussian_count':len(original_local.xyz),'camera':camera.record(),
        'evaluation_camera':ref['camera'].record(),'resolution':cfg['refine_resolution'],
        'settings':cfg,'space':space,'fixed_parameters_radians_logscale_dx_dz':p0.tolist(),
        'fixed_translation_semantics':'Keep the saved horizontal support-center offsets p[2:4] fixed. The renderer retains the original bottom-pivot placement rule when yaw/scale change; no per-grid translation fit.',
        'objective_definition':'reg.evaluate unchanged: 1*(1-soft_iou) + 0.18*appearance_loss + 0.25*boundary_loss + 0.35*center_penalty + 8*volume_penalty; original configured RGB structure/chromaticity and Gaussian covariance spatial penalty.',
        'grid':{'yaw_samples':len(yaws),'scale_samples':len(ratios),'total_points':len(rows),
                'yaw_range_degrees':[float(yaws[0]),float(yaws[-1])],
                'ratio_range':[float(ratios[0]),float(ratios[-1])],
                'spacing_degrees':1.5,'spacing_scale_ratio':.01,'all_points_within_native_scale_bounds':True},
        'selected_solution':{'yaw_degrees':yaw0,'scale_over_selected':1.0,'scale':scale0,
                             'saved_objective':search['selected']['objective'],**selected_metrics},
        'slice_minimum':best,'selected_is_grid_minimum':best['scale_index']==selected_row['scale_index'] and best['yaw_index']==selected_row['yaw_index'],
        'no_interpolation':True,'no_gradient_updates':True,'no_parameter_projection':True,
        'no_registration_rerun':True,'no_m1_m3_or_m4_training':True,'seconds':time.monotonic()-start,
        'runtime':{'gpu':args.gpu,'gpu_name':torch.cuda.get_device_name(0),'torch':torch.__version__,
                   'cuda':torch.version.cuda,'numpy':np.__version__},'script_sha256':sha256(__file__)}
    write_json(out/'grid_summary.json',summary)
    for p in [transform_path,search_path,source/'C/config.yaml',ctx.stage('m2')/'anchor_camera.json',ctx.stage('m2')/'generation_region.json']:
        shutil.copy2(p,out/'records'/p.name)
    shutil.copy2(__file__,out/'code'/Path(__file__).name)
    shutil.copy2(root/'industrial_swap_v2/core/gaussian_registration.py',out/'code/gaussian_registration.py')
    print('SELECTED',summary['selected_solution'],flush=True)
    print('SLICE_MINIMUM',best,flush=True)
    print('GRID COMPLETE',out,flush=True)


if __name__=='__main__':main()
