"""Parameterize the existing GSLE/LangSplat localization rule for fresh instances.

No manual box coordinates or existing target segmentation are used to localize.
The reference SAM segment disambiguates instances; the trained 3D text field is
mandatory. The quantiles, visibility and floor-preservation rule match GSLE M1.
"""
import argparse
import os
from pathlib import Path
import sys
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from ..core.context import read_json,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--survey',default='survey');parser.add_argument('--candidates',default='candidates');args=parser.parse_args()
    root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');base=root/'industrial_swap_v2';batch=args.batch.resolve();batch.relative_to(base)
    os.environ.update(ROOT=str(root),GSLE_OUTPUT=str(batch/'adapters'),GSLE_RECORDS=str(batch/'records'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(root/'gsle'));import common as c
    from world_up import estimate_up
    survey=batch/args.survey;dest=batch/args.candidates;dest.mkdir(exist_ok=True)
    queries=read_json(survey/'queries.json');fields=[np.load(survey/f'scores_level_{i}.npy',mmap_mode='r') for i in (1,2,3)]
    source=c.GaussianData.load(root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply')
    xyz=source.xyz.cpu().numpy();scales=source.log_scale.exp().cpu().numpy();opacity=source.opacity_logit.sigmoid().cpu().numpy()[:,0]
    cameras={cam.name:cam for cam in c.source_cameras(938)}
    floor_scores=np.max(np.stack([f[:,-1] for f in fields]),axis=0)
    up,up_record=estimate_up(xyz,scales,opacity,floor_scores,list(cameras.values()));write_json(batch/'world_up.json',up_record)
    all_records=[];cached_projection={}
    for qi,query in enumerate(queries):
        proposals=read_json(survey/(query['id']+'_top.json'))
        for rank,proposal in enumerate(proposals):
            identifier=f"{query['id']}_{rank:02d}";folder=dest/identifier;folder.mkdir(exist_ok=True)
            if proposal['frame'] not in cameras:
                all_records.append({'id':identifier,'error':'reference camera not reconstructed'});continue
            cam=cameras[proposal['frame']];semantic=np.asarray(fields[proposal['level']-1][:,qi])
            segments=np.load(root/'data/gsle_langsplat/language_features'/(Path(cam.name).stem+'_s.npy'))
            reference=segments[proposal['level']]==proposal['feature_index']
            if reference.shape!=(cam.height,cam.width):raise ValueError('Reference segmentation/camera mismatch')
            xy,z=cam.project(xyz);pixels=np.round(xy).astype(np.int64)
            inside=(z>.01)&(pixels[:,0]>=0)&(pixels[:,0]<cam.width)&(pixels[:,1]>=0)&(pixels[:,1]<cam.height)
            sx=np.clip(pixels[:,0],0,cam.width-1);sy=np.clip(pixels[:,1],0,cam.height-1)
            with torch.no_grad(): rendered=c.render(cam,source)
            distance=np.linalg.norm(xyz-cam.center,axis=-1)
            depth=(rendered['depth']/rendered['alpha'].clamp_min(1e-6))[0].cpu().numpy();local=depth[sy,sx]
            tolerance=np.maximum(scales.max(1)*3,.02*np.median(local[inside]))
            visible=inside&(abs(distance-local)<tolerance)
            seed=visible&reference[sy,sx]&(semantic>.5)&(opacity>.05)
            record={'id':identifier,'query_id':query['id'],'query_index':qi,'proposal':proposal,'seed_count':int(seed.sum())}
            if seed.sum()<=100:
                record.update(eligible=False,reason='insufficient 3D language support');all_records.append(record);write_json(folder/'quality.json',record);continue
            low,high=np.quantile(xyz[seed],[.005,.995],axis=0);margin=(high-low)*.05;low-=margin;high+=margin
            box=((xyz>=low)&(xyz<=high)).all(1);preserved=0
            if 'offset' in up_record:
                floor=(abs(xyz@up+up_record['offset'])<up_record['tolerance'])&(floor_scores>semantic)
                preserved=int((box&floor).sum());box&=~floor
            ids=np.flatnonzero(box)
            if not 100<len(ids)<len(xyz)*.2:
                record.update(eligible=False,reason='unreasonable localization extent',count=len(ids));all_records.append(record);write_json(folder/'quality.json',record);continue
            flags=torch.zeros((len(xyz),3),device='cuda');flags[torch.as_tensor(ids,device='cuda')]=1
            with torch.no_grad():
                contribution=c.raster(cam,source,flags)[0].cpu().numpy()
                isolated=c.render(cam,source.subset(ids))
            mask=contribution>.05;alone=isolated['alpha'][0].cpu().numpy()>.3
            iou=float((alone&reference).sum()/max((alone|reference).sum(),1))
            spill=float((mask&~reference).sum()/max(mask.sum(),1))
            reference_recall=float((mask&reference).sum()/max(reference.sum(),1))
            target={'text':query['text'],'reference_camera':cam.record(),'reference_proposal':proposal,
                'selected_level':proposal['level'],'seed_count':int(seed.sum()),'target_gaussians':len(ids),
                'retained_gaussians':len(xyz)-len(ids),'bbox_low':low.tolist(),'bbox_high':high.tolist(),
                'bbox_corners':c.bbox_corners(low,high).tolist(),'center':((low+high)/2).tolist(),'up':up.tolist(),
                'up_estimation':up_record,'source_ply':str(root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'),
                'source_indices_preserved':True,'manual_roi_used':False,'fresh_semantic_query':True,
                'extent_rule':'0.5%/99.5% quantiles of visible text-positive 3D seeds, plus 5% per-axis margin',
                'removal_rule':'Gaussian centers inside semantic-derived box; preserve coplanar floor-semantic Gaussians',
                'preserved_floor_gaussians':preserved,'floor_preservation_applied':'offset' in up_record,'semantic_boundary':.5}
            write_json(folder/'target.json',target);np.save(folder/'target_indices.npy',ids);np.save(folder/'seed_indices.npy',np.flatnonzero(seed))
            Image.fromarray(reference.astype(np.uint8)*255).save(folder/'reference_mask.png');Image.fromarray(mask.astype(np.uint8)*255).save(folder/'visible_mask.png')
            rgb=(rendered['rgb'].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
            overlay=rgb.copy();overlay[mask]=.4*overlay[mask]+.6*np.array([255,30,30])
            photo=np.asarray(Image.open(root/'data/industrial/images'/cam.name).convert('RGB').resize((cam.width,cam.height))).copy()
            photo[reference]=.4*photo[reference]+.6*np.array([255,30,30])
            board=Image.new('RGB',(938*2,528+32),'white');draw=ImageDraw.Draw(board)
            board.paste(Image.fromarray(photo),(0,32));board.paste(Image.fromarray(overlay),(938,32))
            draw.text((5,5),f"{identifier}: original-photo automatic SAM proposal",fill='black')
            draw.text((943,5),f"Fresh 3D language localization: {len(ids)} Gaussians; IoU={iou:.3f}; spill={spill:.3f}",fill='black')
            board.save(folder/'review.jpg',quality=90)
            eligible=iou>=.45 and spill<=.4 and reference_recall>=.65
            record.update(eligible=bool(eligible),count=len(ids),reference_iou=iou,projection_spill=spill,
                reference_recall=reference_recall,semantic_positive_fraction=float((semantic[ids]>.5).mean()),
                center=target['center'],bbox_low=target['bbox_low'],bbox_high=target['bbox_high'],
                score=float(proposal['semantic_score']*iou*(1-spill)),folder=str(folder))
            write_json(folder/'quality.json',record);all_records.append(record)
            print(identifier,len(ids),'IoU',round(iou,3),'spill',round(spill,3),'eligible',eligible,flush=True)
            del flags,isolated,rendered
        write_json(dest/'candidate_quality.json',all_records)
    print('Candidate localization complete; eligible',sum(r.get('eligible',False) for r in all_records),flush=True)


if __name__=='__main__':main()
