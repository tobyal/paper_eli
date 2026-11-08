"""AVP fallback on calibrated observed cameras when an orbit has no valid view.

Preserves AVP brightness scoring and visibility thresholds. Only the candidate
camera pool changes; calibrated crop intrinsics retain the original pose.
"""
from pathlib import Path
import numpy as np
import torch
from .context import write_json


def select_observed_anchor(c,out,region,ids,settings,brightness_scores,expand_canvas=False,allow_review=False):
    scene=c.GaussianData.load(c.ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply')
    obj=scene.subset(ids);flags=torch.zeros(len(scene.xyz),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
    corners=np.asarray(region['bbox_corners']);records=[];prefix='expanded_observed' if expand_canvas else 'observed';folder=out/'avp'/f'{prefix}_candidates'
    with torch.no_grad():
        for source in c.source_cameras(938):
            xy,z=source.project(corners)
            if not (z>.01).all():continue
            lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2;span=float((hi-lo).max())
            size=int(np.ceil(max(span*(1.35 if expand_canvas else 2.2),source.height*.4)))
            if size>(2*source.width if expand_canvas else source.height) or size<32:continue
            left=int(round(center[0]-size/2)) if expand_canvas else int(np.clip(round(center[0]-size/2),0,source.width-size))
            top=int(round(center[1]-size/2)) if expand_canvas else int(np.clip(round(center[1]-size/2),0,source.height-size))
            if not (lo>np.array([left,top])+4).all() or not (hi<np.array([left+size,top+size])-4).all():continue
            factor=512/size
            cam=c.Camera(source.R,source.t,512,512,source.fx*factor,source.fy*factor,
                (source.cx-left)*factor,(source.cy-top)*factor,prefix+'_'+Path(source.name).stem)
            joined=c.render(cam,scene,flags);isolated=c.render(cam,obj)
            mask=(isolated['alpha'][0]>.5).cpu().numpy();rgb=isolated['rgb'].permute(1,2,0).cpu().numpy()
            score=brightness_scores(rgb,mask)
            visible=float(joined['object_alpha'].sum()/isolated['alpha'].sum().clamp_min(1e-6))
            coverage=float((joined['alpha']>.5).float().mean())
            eligible=score is not None and visible>=settings['min_visibility'] and coverage>=settings['min_scene_coverage']
            record={'index':len(records),'source_camera':source.name,'camera':cam.record(),'visibility':visible,
                'scene_coverage':coverage,'score':score,'eligible':eligible,'source_crop_xyxy':[left,top,left+size,top+size],
                'source_resolution':[source.width,source.height],'candidate_pool':'calibrated_observed_expanded_canvas' if expand_canvas else 'calibrated_observed_cameras'}
            records.append(record)
            c.save_image(folder/(Path(source.name).stem+'.png'),joined['rgb'])
        write_json(out/'avp'/f'{prefix}_candidates.json',records)
        eligible=[r for r in records if r['eligible']]
        if not eligible and not allow_review:raise RuntimeError('Neither orbit nor observed cameras meet unchanged AVP visibility/coverage thresholds')
        review=[r for r in records if r['score'] is not None and r['scene_coverage']>=settings['min_scene_coverage']]
        if not eligible and not review:raise RuntimeError('No context-covered observed camera for review')
        chosen=max(eligible,key=lambda r:r['score']['text_contrast']) if eligible else max(review,key=lambda r:r['visibility'])
        metadata={'selected':chosen,'camera':chosen['camera'],'scoring':'text_contrast' if eligible else 'visibility_first_manual_review_only','eligible_views':len(eligible),
            'quality_status':'passed' if eligible else 'manual_review_required',
            'quality_note':None if eligible else 'Best observed-pose visibility is below the unchanged threshold. Human review required; not AVP-eligible.',
            'candidate_pool':chosen['candidate_pool'],'adaptation':('Original and sensor-bounded observed pools have no eligible view. Preserve calibrated R/t and render a target-centered square canvas, including rays outside the original photo bounds, to contain the entire editing volume with 35% context margin. This is a new Gaussian render, not image padding or a forced M1 reference view.' if expand_canvas else 'Original 100-view orbit had no eligible view. Same AVP brightness score and thresholds on calibrated observed poses with centered square crops.'),
            'thresholds':settings}
        write_json(out/'avp/anchor.json',metadata);write_json(out/'records/avp.json',metadata)
    del scene,obj,flags;torch.cuda.empty_cache()
    print('Selected observed-camera AVP fallback',chosen['source_camera'],'visibility',chosen['visibility'],flush=True)
