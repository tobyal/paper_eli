"""Case02: genuine coarse / legacy fit / rasterized fit, without asset training.

Run on the server from the repository root with envs/gsle/bin/python.
Original experiment directories are read-only inputs. Each solver is called
unchanged, and the new solver explicitly has no historical transform seed.
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), SWAP_DISABLE_QWEN='1',
                      OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    import cv2
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from industrial_swap_v2.core.context import Context, gsle, read_json, write_json, sha256
    from industrial_swap_v2.core import gaussian_registration as reg
    from industrial_swap_v2.core.alignment import fit, transform_vertices, xyz_of
    from industrial_swap_v2.core.m2_space import registration_space
    from industrial_swap_v2.core.scene import read_vertices, save_vertices, compose
    from industrial_swap_v2.core.pbr_gaussian_optimizer import render

    base = root/'industrial_swap_v2/outputs/case02_raster_alignment_v1'
    dest = root/'industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration'
    if dest.exists() and not args.evaluate_only:
        raise FileExistsError('Preserve existing experiment: '+str(dest))
    dest.mkdir(parents=True, exist_ok=True)
    panels = dest/'panels'; panels.mkdir(exist_ok=True)
    code = dest/'code'; code.mkdir(exist_ok=True)
    cfg = yaml.safe_load((base/'config.yaml').read_text())
    cfg['project']['gpu'] = args.gpu
    cfg['m4']['integration'] = {'alignment': 'rasterized_reference', 'registration': dict(reg.DEFAULTS)}
    cfg['m2'].pop('api_key_file', None)
    names = [('A', 'Coarse Placement'), ('B', 'Legacy Registration'), ('C', 'Gaussian-Aware Registration')]
    contexts = {}
    for key, name in names:
        run = dest/key; run.mkdir(exist_ok=True)
        cc = copy.deepcopy(cfg)
        for k, v in [('output_dir', ''), ('records_dir', 'records'), ('logs_dir', 'logs')]:
            cc['project'][k] = str((run/v).relative_to(root/'industrial_swap_v2'))
        for stage in ('m1','m2','m3'):
            link = run/stage
            if not link.exists(): link.symlink_to((base/stage).resolve(), target_is_directory=True)
        (run/'config.yaml').write_text(yaml.safe_dump(cc,sort_keys=False))
        contexts[key] = Context(run/'config.yaml')
    ctx = contexts['A']; common = gsle(ctx)
    torch.manual_seed(42); np.random.seed(42)
    torch.cuda.set_per_process_memory_fraction(.5)
    sources = [base/'config.yaml', base/'m3/object_refined_local.ply',
               base/'m2/anchor_camera.json',base/'m2/object_mask.png',base/'m2/object_rgba.png',
               base/'m2/generation_region.json',base/'m2/conditions.json',
               base/'m1/background_gaussians.ply',base/'m1/manifest.json',base/'m1/edit_region.json']
    source_records = [{'path':str(p.resolve()),'sha256':sha256(p)} for p in sources]
    paths = ['core/alignment.py','core/gaussian_registration.py','core/m2_space.py',
             'core/scene_integration.py','core/pbr_gaussian_optimizer.py','core/pbr_math.py']
    code_records = []
    for rel in paths:
        p = root/'industrial_swap_v2'/rel
        shutil.copy2(p, code/p.name)
        code_records.append({'path':str(p),'sha256':sha256(p)})
    shutil.copy2(Path(__file__), code/Path(__file__).name)
    space = registration_space(ctx); frame = space['frame']
    camera_record = read_json(ctx.stage('m2')/'anchor_camera.json')
    camera = common.Camera.from_record(camera_record)
    write_json(dest/'fixed_inputs.json', {'sources':source_records,'code':code_records,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'space':space,'camera':camera_record,'seed':42,'gpu':args.gpu,
        'independence':'B calls legacy fit; C starts from its native yaw/size pool. No historical transform or B result is supplied to C.',
        'no_asset_training':True,'no_background_optimization':True})
    asset_path = ctx.stage('m3')/'object_refined_local.ply'
    vertices = read_vertices(asset_path)
    timings = {}
    if not args.evaluate_only:
        start = time.monotonic()
        obj = common.GaussianData.load(asset_path)
        model = reg.Pose(common,obj,space,dict(reg.DEFAULTS))
        initial = model.tensor([0,math.log(model.base_scale),0,0])
        transform = reg.pose_report(model,initial)
        transform.update(method='Coarse Placement', initialization='Zero local yaw; uniform robust Gaussian-extent fit to M2 box; measured M1 footprint center and fixed support height; no image fitting.')
        out = ctx.stage('m4')
        save_vertices(out/'object_world.ply',transform_vertices(vertices,transform['scale'],
            np.asarray(transform['rotation']),np.asarray(transform['translation'])))
        write_json(out/'transform.json',transform)
        timings['A'] = time.monotonic()-start
        del model,obj;gc.collect();torch.cuda.empty_cache()
        print('A COMPLETE',timings['A'],flush=True)
        start = time.monotonic()
        fit(vertices,frame,camera_record,ctx.stage('m2')/'object_mask.png',cfg['m3'],contexts['B'].stage('m4'))
        timings['B'] = time.monotonic()-start
        print('B LEGACY FIT COMPLETE',timings['B'],flush=True)
        write_json(dest/'timing.json',timings)
        start = time.monotonic()
        reg.fit_rasterized(contexts['C'])
        timings['C'] = time.monotonic()-start
        write_json(dest/'timing.json',timings)
        print('C RASTERIZED FIT COMPLETE',timings['C'],flush=True)
    else:
        timings = read_json(dest/'timing.json')

    # Evaluate exported PLYs (including rotated SH), not the solvers' objectives.
    ref = reg.reference(ctx,camera,1024,'cuda'); eval_cam=ref['camera']
    common.save_image(panels/'reference_white.png',ref['rgb'])
    truth=(ref['mask'].cpu().numpy()>.5).astype(np.uint8)
    Image.fromarray(truth*255).save(panels/'reference_mask.png')
    contours,_=cv2.findContours(truth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    origin=np.asarray(frame['center']); axes=np.asarray(frame['axes'])
    low=np.asarray(frame['local_bbox_low']); high=np.asarray(frame['local_bbox_high'])
    background=common.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    records=[];transforms={};clouds={};bounds=[]
    with torch.no_grad():
        common.save_image(panels/'background_anchor.png',common.render(eval_cam,background,aux=False)['rgb'])
        for key,name in names:
            run=contexts[key].stage('m4'); path=run/'object_world.ply'
            actual=read_vertices(path);data=common.GaussianData.load(path)
            assert len(actual)==len(vertices)
            assert actual['opacity'].tobytes()==vertices['opacity'].tobytes()
            pred=render(common,eval_cam,data);metrics=reg.numbers(reg.image_metrics(pred,ref))
            transform=read_json(run/'transform.json');rel_rot=axes.T@np.asarray(transform['rotation'])
            local=(xyz_of(actual)-origin)@axes
            outside=((local<low)|(local>high)).any(1)
            # Auxiliary all-Gaussian covariance support metric (not the primary center-count metric).
            quat=np.column_stack([actual[f'rot_{i}'] for i in range(4)])[:,[1,2,3,0]]
            rot=Rotation.from_quat(quat).as_matrix()
            scales=np.exp(np.column_stack([actual[f'scale_{i}'] for i in range(3)]))
            local_rot=np.einsum('ij,njk->nik',axes.T,rot)
            sigma=np.sqrt(((local_rot*scales[:,None,:])**2).sum(2))
            support_outside=((local-sigma<low)|(local+sigma>high)).any(1)
            row={'method':key,'name':name,'silhouette_iou':metrics['mask_iou'],
                 'reference_rgb_error':metrics['foreground_rgb_mae'],
                 'workspace_overflow':float(outside.mean()),
                 'workspace_overflow_count':int(outside.sum()),'gaussian_count':len(actual),
                 'support_1sigma_outside_fraction':float(support_outside.mean()),
                 'scale':float(transform['scale']),
                 'yaw_degrees':float(np.degrees(np.arctan2(rel_rot[0,2],rel_rot[0,0])))%360,
                 'translation_x':transform['translation'][0],'translation_y':transform['translation'][1],
                 'translation_z':transform['translation'][2],
                 'center_bottom_gap':float(local[:,1].min()-low[1]),
                 'seconds':timings[key],'opacity_unchanged':True}
            records.append(row); transforms[key]={'name':name,**transform}
            clouds[key]=local.astype(np.float32)
            np.save(run/'gaussian_centers_generation_frame.npy',local)
            write_json(run/'unified_evaluation.json',{'primary':row,'renderer_metrics':metrics,
                'local_center_bounds':[local.min(0).tolist(),local.max(0).tolist()],
                'PLY_sha256':sha256(path)})
            common.save_image(panels/f'{key}_object_rgb.png',pred['rgb'])
            Image.fromarray(np.round(pred['alpha'].cpu().numpy().clip(0,1)*255).astype(np.uint8)).save(panels/f'{key}_object_alpha.png')
            np.save(panels/f'{key}_object_depth.npy',pred['depth_z'].cpu().numpy())
            scene=common.render_joint(eval_cam,background,data,aux=False)['rgb']
            common.save_image(panels/f'{key}_scene_full.png',scene)
            arr=np.asarray(Image.open(panels/f'{key}_scene_full.png')).copy()
            cv2.drawContours(arr,contours,-1,(0,220,230),2,cv2.LINE_AA)
            Image.fromarray(arr).save(panels/f'{key}_scene_contour_full.png')
            mask=pred['alpha'].cpu().numpy()>.5
            ys,xs=np.where(mask|truth.astype(bool));bounds.append([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
            del data,pred,scene;gc.collect();torch.cuda.empty_cache()
            print('UNIFIED EVALUATION',row,flush=True)
    # One shared crop covers every method and reference; never method-specific framing.
    b=np.asarray(bounds);xy0=b[:,:2].min(0);xy1=b[:,2:].max(0)
    mid=(xy0+xy1)/2;size=(xy1-xy0)*1.20;size=np.maximum(size,size.max()*.82)
    crop=[max(0,int(mid[0]-size[0]/2)),max(0,int(mid[1]-size[1]/2)),
          min(eval_cam.width,int(mid[0]+size[0]/2)),min(eval_cam.height,int(mid[1]+size[1]/2))]
    for key,_ in names:
        Image.open(panels/f'{key}_scene_contour_full.png').crop(crop).save(panels/f'{key}_anchor.png')
    np.savez_compressed(dest/'workspace_points.npz',**clouds,low=low,high=high,axes=axes,origin=origin)
    write_json(dest/'registration_transforms.json',transforms)
    with (dest/'registration_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    preserved={s['path']:sha256(s['path'])==s['sha256'] for s in source_records}
    assert all(preserved.values())
    report={'case':'case02','inputs':source_records,'code':code_records,
        'fixed_inputs':'fixed_inputs.json','camera':camera_record,'evaluation_camera':eval_cam.record(),
        'crop_xyxy':crop,'alpha_threshold':.5,'reference_mask_threshold':.5,
        'silhouette_iou':'Binary IoU at 1024x1024, predicted alpha > 0.5 and resized reference > 0.5.',
        'reference_rgb_error':'Mean absolute display-RGB error [0,1], white-composited object render vs white-composited M2 RGBA, weighted by the resized reference mask, averaged over 3 channels. No color/exposure fit.',
        'workspace_overflow':'Unweighted fraction of ALL exported Gaussian centers outside the identical oriented M2 generation box; no opacity threshold or quantile trimming. Covariance support fraction additionally reported in CSV.',
        'footprints':'Top-down convex envelopes of all actual transformed Gaussian centers in the common M2 frame; not fitted boxes.',
        'independent_solves':True,'existing_transform_removed_from_C':True,
        'shared_background':str((ctx.stage('m1')/'background_gaussians.ply').resolve()),
        'source_preservation':preserved,'metrics':records,'background_optimization':False}
    write_json(dest/'sources.json',report)
    print('F13 EXPERIMENT COMPLETE',dest,flush=True)


if __name__=='__main__':
    main()
