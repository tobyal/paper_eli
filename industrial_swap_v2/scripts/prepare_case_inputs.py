"""Prepare approved fresh localization candidates through M2 conditions only.

The only invoked stages are existing M1 partitioning, existing AVP and M2
prepare_only. Qwen is disabled and its generate function is also replaced by
an explicit error for this process. M3/M4 are never imported or invoked.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from ..core.context import Context,read_json,write_json,sha256


def region_from_target(c,target):
    up=np.asarray(target['up']);center=np.asarray(target['center']);ref=c.Camera.from_record(target['reference_camera'])
    toward=ref.center-center;front=c.normalized(toward-up*np.dot(toward,up));right=c.normalized(np.cross(up,front))
    axes=np.stack((right,up,front),axis=1);local=np.asarray(target['bbox_corners'])@axes
    low=local.min(0);high=local.max(0);sides=high-low;high[1]=low[1]+sides.max()
    corners=c.bbox_corners(low,high)@axes.T
    return {**target,'center':(((low+high)/2)@axes.T).tolist(),'bbox_corners':corners.tolist(),
        'bbox_low':corners.min(0).tolist(),'bbox_high':corners.max(0).tolist(),
        'object_scale':float((high-low).max()),'region_axes':axes.tolist(),
        'original_bbox_corners':target['bbox_corners'],'original_oriented_side_lengths':sides.tolist(),
        'generation_oriented_side_lengths':(high-low).tolist(),
        'generation_region_rule':'same gsle/prepare_edit_region.py: preserve original horizontal footprint and bottom; height=max original extent',
        'manual_coordinates':False}


def prepare_one(batch,entry,c,defaults):
    root=c.ROOT;base=root/'industrial_swap_v2';out=batch/entry['case_id']
    candidate=batch/entry['candidate'];target=read_json(candidate/'target.json');quality=read_json(candidate/'quality.json')
    if (out/'case.json').exists():
        if read_json(out/'case.json')['candidate']!=entry['candidate']:raise ValueError('Existing case uses a different candidate')
        print('Preserving prepared case',entry['case_id'],flush=True);return
    out.mkdir(exist_ok=True);localization=out/'localization';localization.mkdir(exist_ok=True);records=out/'records';records.mkdir(exist_ok=True)
    if (localization/'target_indices.npy').exists() and sha256(localization/'target_indices.npy')!=sha256(candidate/'target_indices.npy'):
        raise ValueError('Partial case uses different localization; use a new directory')
    if not quality['eligible']:raise ValueError('Candidate failed numerical localization checks')
    for name in ('target_indices.npy','seed_indices.npy','reference_mask.png','visible_mask.png','review.jpg','quality.json'):
        shutil.copy2(candidate/name,localization/name)
    ids=np.load(localization/'target_indices.npy');count=defaults['source_count']
    keep=np.ones(count,bool);keep[ids]=False;np.save(localization/'keep_indices.npy',np.flatnonzero(keep))
    write_json(localization/'target.json',target);region=region_from_target(c,target);write_json(localization/'edit_region.json',region)
    import avp
    avp.OUT=out;avp.editing_region=lambda:region
    avp_config=copy.deepcopy(defaults['gsle_config'])
    avp_config['avp']['min_visibility']=entry.get('avp_min_visibility',avp_config['avp']['min_visibility'])
    write_json(out/'avp_settings.json',avp_config['avp'])
    avp.config=lambda:avp_config
    def save_avp(path,value):
        path=Path(path)
        if path==root/'records/gsle/avp.json':path=records/'avp.json'
        path.resolve().relative_to(out.resolve());write_json(path,value)
    avp.save_json=save_avp
    if not (out/'avp/anchor.json').exists():
        candidates_file=out/'avp/candidates.json'
        try:
            if candidates_file.exists():
                eligible=[r for r in read_json(candidates_file) if r['score'] is not None
                    and r['visibility']>=avp_config['avp']['min_visibility']
                    and r['scene_coverage']>=avp_config['avp']['min_scene_coverage']]
                if not eligible:raise RuntimeError('No visible AVP candidate')
                chosen=dict(max(eligible,key=lambda r:r['score']['text_contrast']),eligible=True)
                metadata={'selected':chosen,'camera':chosen['camera'],'scoring':'text_contrast',
                    'eligible_views':len(eligible),'thresholds':avp_config['avp'],
                    'candidate_pool':'same_saved_100_view_orbit','source_candidates_immutable':True}
                write_json(out/'avp/anchor.json',metadata);write_json(records/'avp.json',metadata)
            else:avp.main()
        except RuntimeError as error:
            if 'No visible AVP candidate' not in str(error):raise
            from ..core.observed_anchor import select_observed_anchor
            select_observed_anchor(c,out,region,ids,avp_config['avp'],avp.brightness_scores)
    anchor=read_json(out/'avp/anchor.json')
    cfg=copy.deepcopy(defaults['config']);cfg['project'].update(
        output_dir=str(out.relative_to(base)),records_dir=str(records.relative_to(base)),
        logs_dir=str((out/'logs').relative_to(base)),gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','2')))
    cfg['m1'].update(localization_dir=str(localization.relative_to(root)),target_text=target['text'])
    cfg['m2'].update(existing_anchor=str((out/'avp/anchor.json').relative_to(root)),reuse_anchor=True,
        replacement_text=entry['replacement'],prompt_mode='generic_device',segmentation_prompt_mode='box_clip')
    cfg['m3']['input_image']=str((out/'m2/object_rgba.png').relative_to(base))
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    ctx=Context(out/'config.yaml')
    from ..modules.m1_localization import run as run_m1
    from ..modules import m2_image_generation
    def forbidden(*args,**kwargs):raise RuntimeError('Case preparation forbids all Qwen submissions')
    m2_image_generation.generate=forbidden
    run_m1(ctx);m2_image_generation.run(ctx,prepare_only=True)
    scene=c.GaussianData.load(ctx.source(cfg['project']['source_ply']));bg=scene.subset(np.flatnonzero(keep))
    flags=torch.zeros((len(scene.xyz),3),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    cameras=c.source_cameras(938);reference=c.Camera.from_record(target['reference_camera'])
    neighbors=sorted([cam for cam in cameras if cam.name!=reference.name],
        key=lambda cam:np.linalg.norm(cam.center-reference.center)+np.linalg.norm(cam.R-reference.R))
    review=out/'m1/review_views';reviews=[]
    with torch.no_grad():
        for cam in [camera,reference,*neighbors[:2]]:
            before=c.render(cam,scene);background=c.render(cam,bg)
            contribution=c.raster(cam,scene,flags)[0];mask=contribution>.05
            overlay=before['rgb'].clone();tint=torch.tensor([1.,.1,.05],device='cuda')[:,None]
            overlay[:,mask]=.4*overlay[:,mask]+.6*tint
            directory=review/Path(cam.name).stem
            for name,image in [('rgb',before['rgb']),('old_mask',mask),('overlay',overlay),('background',background['rgb']),
                ('depth_original',c.depth_condition(before['depth'],before['alpha'])),
                ('depth_background_rendered',c.depth_condition(background['depth'],background['alpha']))]:
                c.save_image(directory/(name+'.png'),image)
            write_json(directory/'camera.json',cam.record())
            reviews.append({'camera':cam.record(),'visible_mask_pixels':int(mask.sum()),'folder':str(directory)})
            if cam.name==camera.name:
                shutil.copy2(directory/'old_mask.png',out/'m1/old_device_mask.png')
                shutil.copy2(directory/'overlay.png',out/'m1/localization_overlay.png')
                shutil.copy2(directory/'background.png',out/'m1/background_render.png')
                shutil.copy2(directory/'depth_background_rendered.png',out/'m2/depth_background_rendered.png')
    write_json(out/'case.json',{**entry,'semantic_text':target['text'],'target_count':len(ids),
        'target_indices_sha256':sha256(localization/'target_indices.npy'),'anchor':anchor['selected'],
        'anchor_file':str(out/'m2/anchor_camera.json'),'config':str(out/'config.yaml'),
        'quality':quality,'review_views':reviews,'status':'prepared_for_human_review_before_qwen',
        'qwen_calls':0,'trellis_calls':0,'m3_executed':False,'m4_executed':False,
        'background_depth_definition':'Original rendered depth zeroed inside edit volume; unknown, not filled ground. Removed-background depth additionally saved for inspection.'})
    del scene,bg,flags;torch.cuda.empty_cache()
    print('CASE PREPARED',entry['case_id'],target['text'],len(ids),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--continue-on-error',action='store_true');args=parser.parse_args()
    root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');batch=args.batch.resolve();batch.relative_to(root/'industrial_swap_v2')
    os.environ.update(ROOT=str(root),GSLE_OUTPUT=str(batch/'adapters'),GSLE_RECORDS=str(batch/'records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(root/'gsle'));import common as c
    defaults={'config':yaml.safe_load((root/'industrial_swap_v2/configs/default.yaml').read_text()),
        'gsle_config':read_json(root/'gsle/config.json')}
    from ..core.scene import read_vertices
    defaults['source_count']=len(read_vertices(root/defaults['config']['project']['source_ply']))
    start=time.time()
    failures=[]
    for entry in read_json(args.selection):
        try:prepare_one(batch,entry,c,defaults)
        except Exception as error:
            if not args.continue_on_error:raise
            traceback.print_exc();failures.append({'case_id':entry['case_id'],'error':str(error)})
            torch.cuda.empty_cache()
    write_json(batch/'prepare_completion.json',{'seconds':time.time()-start,'selection':str(args.selection),
        'status':'partial' if failures else 'stopped_before_qwen','failures':failures,
        'qwen_calls':0,'trellis_calls':0,'m3_executed':False,'m4_executed':False})


if __name__=='__main__':main()
