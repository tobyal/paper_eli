"""Archived camera-search diagnostic; novel high-angle views were not used in the final 30 inputs."""
import math
import numpy as np
import torch
from industrial_swap_v2.core.context import read_json,write_json


def select_elevated_anchor(c,out,region,ids,settings,brightness_scores,allow_review=False):
    base=read_json(out/'avp/candidates.json')[0]
    center=np.asarray(region['center']);up=np.asarray(region['up'])
    reference=c.Camera.from_record(region['reference_camera']);toward=reference.center-center
    horizontal=c.normalized(toward-up*np.dot(toward,up));right=c.normalized(np.cross(up,horizontal))
    radius=base['radius'];scene=c.GaussianData.load(c.ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply')
    obj=scene.subset(ids);flags=torch.zeros(len(scene.xyz),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
    existing=out/'avp/elevated_candidates.json'
    records=read_json(existing) if existing.exists() else [];eligible=[r for r in records if r['eligible']]
    with torch.no_grad():
        for elevation in (() if records else (35,50,65)):
            if abs(elevation-base['elevation'])<2:continue
            for i in range(settings['views']):
                angle=2*math.pi*i/settings['views'];e=math.radians(elevation)
                position=center+radius*(math.cos(e)*(math.cos(angle)*horizontal+math.sin(angle)*right)+math.sin(e)*up)
                cam=c.look_at(position,center,up,512,49.1,f'avp_e{elevation}_{i:03d}')
                joined=c.render(cam,scene,flags);isolated=c.render(cam,obj)
                mask=(isolated['alpha'][0]>.5).cpu().numpy();rgb=isolated['rgb'].permute(1,2,0).cpu().numpy()
                score=brightness_scores(rgb,mask)
                visible=float(joined['object_alpha'].sum()/isolated['alpha'].sum().clamp_min(1e-6))
                coverage=float((joined['alpha']>.5).float().mean())
                good=score is not None and visible>=settings['min_visibility'] and coverage>=settings['min_scene_coverage']
                row={'index':len(records),'azimuth':i*360/settings['views'],'elevation':elevation,'radius':radius,
                     'camera':cam.record(),'visibility':visible,'scene_coverage':coverage,'score':score,'eligible':good}
                records.append(row)
                if good:eligible.append(row)
                c.save_image(out/'avp/elevated_candidates'/f'{cam.name}.png',joined['rgb'])
            print('Elevated AVP',elevation,'best visibility',max(r['visibility'] for r in records if r['elevation']==elevation),'eligible so far',len(eligible),flush=True)
        write_json(out/'avp/elevated_candidates.json',records)
        if not eligible and not any('fov' in r for r in records):
            # Deterministic local proposals near the best visibility directions;
            # change FoV with radius to retain the original bounding-sphere margin.
            seeds=sorted(records,key=lambda r:r['visibility'],reverse=True)[:8]
            for seed in seeds:
                for radius_factor in (.75,.9,1.1):
                    for de in (-5,0,5):
                        for da in (-1.8,1.8):
                            elevation=seed['elevation']+de;azimuth=seed['azimuth']+da
                            e=math.radians(elevation);angle=math.radians(azimuth)
                            fov=math.degrees(2*math.asin(math.sin(math.radians(49.1/2))/radius_factor))
                            position=center+radius*radius_factor*(math.cos(e)*(math.cos(angle)*horizontal+math.sin(angle)*right)+math.sin(e)*up)
                            cam=c.look_at(position,center,up,512,fov,f'avp_refine_{len(records):04d}')
                            joined=c.render(cam,scene,flags);isolated=c.render(cam,obj)
                            mask=(isolated['alpha'][0]>.5).cpu().numpy();rgb=isolated['rgb'].permute(1,2,0).cpu().numpy()
                            score=brightness_scores(rgb,mask)
                            visible=float(joined['object_alpha'].sum()/isolated['alpha'].sum().clamp_min(1e-6))
                            coverage=float((joined['alpha']>.5).float().mean())
                            good=score is not None and visible>=settings['min_visibility'] and coverage>=settings['min_scene_coverage']
                            row={'index':len(records),'azimuth':azimuth,'elevation':elevation,'radius':radius*radius_factor,
                                 'fov':fov,'camera':cam.record(),'visibility':visible,'scene_coverage':coverage,'score':score,'eligible':good}
                            records.append(row)
                            if good:eligible.append(row)
                            c.save_image(out/'avp/elevated_candidates'/f'{cam.name}.png',joined['rgb'])
                print('Refined AVP best visibility',max(r['visibility'] for r in records),'eligible',len(eligible),flush=True)
            write_json(out/'avp/elevated_candidates.json',records)
        if not eligible and not allow_review:raise RuntimeError('Expanded AVP elevations also fail unchanged visibility/coverage thresholds')
        review=[r for r in records if r['score'] is not None and r['scene_coverage']>=settings['min_scene_coverage']]
        if not eligible and not review:raise RuntimeError('No usable context-covered candidate even for manual review')
        chosen=max(eligible,key=lambda r:r['score']['text_contrast']) if eligible else max(review,key=lambda r:r['visibility'])
        metadata={'selected':chosen,'camera':chosen['camera'],'scoring':'text_contrast' if eligible else 'visibility_first_manual_review_only','eligible_views':len(eligible),
            'quality_status':'passed' if eligible else 'manual_review_required',
            'quality_note':None if eligible else 'Best available candidate is below the unchanged visibility threshold; prepared for human inspection, not marked AVP-eligible.',
            'candidate_pool':'extended_elevation_orbit','thresholds':settings,
            'adaptation':'No valid original-orbit or calibrated-observed candidate. Added 100 azimuths at each of 35/50/65 degrees elevation. If needed, refine around the eight most visible directions with azimuth offsets +/-1.8 deg, elevation offsets -5/0/+5 deg and radius factors .75/.9/1.1; adapt FoV to retain the original bounding-sphere margin. Preserve M1 geometry, original center, brightness scoring and visibility/coverage thresholds.'}
        write_json(out/'avp/anchor.json',metadata);write_json(out/'records/avp.json',metadata)
    del scene,obj,flags;torch.cuda.empty_cache()
    print('Selected elevated AVP',chosen['camera']['name'],'visibility',chosen['visibility'],flush=True)
