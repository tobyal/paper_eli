"""Read-only paper panels from saved PLYs. No generation or optimization."""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
sys.path.insert(0, str(ROOT))
from baseline_comparison.matched_rendering import render as r
c = r.c
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v3'
ORIGINAL = ROOT / 'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
CASES = r.read(ROOT / 'baseline_comparison/prompts.json')['cases']


def source(path):
    p = Path(path)
    return dict(path=str(p), resolved=str(p.resolve()), sha256=r.sha(p), bytes=p.stat().st_size)


def rgb(path, cam, data, white=False):
    c.save_image(path, c.render(cam, data, aux=False,
        background=torch.ones(3, device='cuda') if white else None)['rgb'])


def final_object(row):
    return Path(row['ours_scene']).parent / 'object_world.ply'


def accepted_cam(case, view='main'):
    rec = r.read(ROOT / f'baseline_comparison/comparison_updated_v8/{case}/camera_params.json')
    return c.Camera.from_record(rec['methods']['ours'][view]['final_camera'])


def qwen_image(row):
    if row['case'] == 'case02':
        p = ROOT / 'industrial_swap_v2/outputs/qwen9_completion_20261107/inputs/case_02_B.png'
        assert r.sha(p) == 'b794ff4b999f7d2a3c7d2a46a1064aceaec784092146ab038874e5a3c8d4baac'
        return p
    return Path(row['ours_anchor']).parent / 'edited_scene.png'


def contact(images, dest, columns=3, size=360):
    canvas = Image.new('RGB', (columns*size, int(np.ceil(len(images)/columns))*(size+26)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for i,(label,path) in enumerate(images):
        im = ImageOps.contain(Image.open(path).convert('RGB'), (size,size))
        x=(i%columns)*size; y=(i//columns)*(size+26)
        canvas.paste(im,(x+(size-im.width)//2,y+(size-im.height)//2))
        draw.text((x+6,y+size+3),label,fill='black',font=font)
    canvas.save(dest)


@torch.no_grad()
def f05():
    row=CASES[0];dest=OUT/'F05_case00';pan=dest/'panels';pan.mkdir(parents=True,exist_ok=True)
    assert not (dest/'sources.json').exists(), 'Completed figure exists; preserve it'
    photo=ROOT/'data/industrial/images'/row['reference_image']
    im=Image.open(photo).convert('RGB');mask=np.load(row['reference_sam_mask'])
    maskim=Image.fromarray((mask.astype(bool)*255).astype(np.uint8)).resize(im.size,Image.Resampling.NEAREST)
    arr=np.asarray(im).astype(float);a=np.asarray(maskim)>0
    arr[a]=.6*arr[a]+.4*np.array([255,35,35])
    Image.fromarray(np.uint8(np.round(arr))).save(pan/'a_registered_reference.png')
    shutil.copy2(photo,pan/'reference_photo_original.jpg');maskim.save(pan/'reference_mask.png')
    cam=accepted_cam('case00').resize(1600,1600)
    original=c.GaussianData.load(ORIGINAL)
    rgb(pan/'d_original_3dgs.png',cam,original)
    idxpath=Path(row['target_record']).parent/'target_indices.npy';idx=np.load(idxpath)
    colors=original.colors(cam.center);idx=torch.as_tensor(idx,device='cuda')
    colors[idx]=.4*colors[idx]+.6*torch.tensor([1.,0.,0.],device='cuda')
    c.save_image(pan/'b_target_localization.png',c.raster(cam,original,colors))
    del original,colors,idx;gc.collect();torch.cuda.empty_cache()
    shutil.copy2(qwen_image(row),pan/'c_qwen_reference.png')
    final=c.GaussianData.load(row['ours_scene'])
    rgb(pan/'e_reconfigured_scene.png',cam,final)
    novel=accepted_cam('case00','aux1').resize(1600,1600)
    rgb(pan/'f_novel_view.png',novel,final)
    obj=c.GaussianData.load(final_object(row))
    details={}
    for name,cc in [('e_reconfigured_scene',cam),('f_novel_view',novel)]:
        box=r.alpha_stats(r.object_alpha(cc,obj))['bbox'];crop=r.square_crop(box,1600,padding=1.25)
        Image.open(pan/(name+'.png')).crop(crop).save(pan/(name+'_detail.png'))
        details[name]=crop
    sources=dict(case='case00',old_text=row['A'],new_text=row['B'],original_scene=source(ORIGINAL),
        final_scene=source(row['ours_scene']),final_object=source(final_object(row)),
        registered_photo=source(photo),reference_sam_mask=source(row['reference_sam_mask']),
        m1_target=source(row['target_record']),localization_indices=source(idxpath),
        qwen=source(qwen_image(row)),reference_camera=row['reference_camera'],
        main_camera=cam.record(),novel_camera=novel.record(),detail_crops=details,
        local_background_iterations=2000,previous_automatic_guard_passed=row.get('ours_guard_passed'),
        annotation='Photo: 40% red SAM overlay. Localization: actual target Gaussian colors blended with 60% red before joint rasterization.',
        original_and_final_same_camera=True,source_ply_modified=False)
    r.save(dest/'sources.json',sources)
    contact([(x.name,x) for x in sorted(pan.glob('[a-f]_*.png')) if '_detail' not in x.name],dest/'panels_preview.jpg')
    print('F05 PANELS COMPLETE',flush=True)


@torch.no_grad()
def f06_assets():
    """Saved PBR camera candidates only; choose a presentation view, not model parameters."""
    dest=OUT/'F06_five_cases';dest.mkdir(parents=True,exist_ok=True)
    records={};images=[]
    for row in CASES:
        case=row['case'];p=Path(row['ours_source'])/'m3/object_refined_local.ply'
        cameras_path=p.parent/'pbr_views/cameras.json';data=r.read(cameras_path)
        asset=c.GaussianData.load(p);cd=dest/'asset_view_candidates'/case;cd.mkdir(parents=True,exist_ok=True)
        cameras=[]
        for rec in data['views']:
            if rec['elevation_degrees']!=30 or rec['azimuth_degrees']%60:continue
            cam=c.Camera.from_record(rec).resize(360,360)
            path=cd/(rec['name']+'.png');rgb(path,cam,asset,True)
            cameras.append(cam.record());images.append((case+' az '+str(rec['azimuth_degrees']),path))
        records[case]=dict(cameras=cameras,asset=source(p),camera_source=source(cameras_path))
        del asset;gc.collect();torch.cuda.empty_cache()
    r.save(dest/'asset_candidates.json',records)
    contact(images,dest/'asset_candidates.jpg',columns=6,size=300)
    print('F06 ASSET CAMERAS COMPLETE',flush=True)


@torch.no_grad()
def f06(choices):
    dest=OUT/'F06_five_cases';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'case_manifest.json').exists()
    manifest=[];images=[]
    for row in CASES:
        case=row['case'];pan=dest/'panels'/case;pan.mkdir(parents=True,exist_ok=True)
        cam=c.Camera.from_record(r.read(row['ours_anchor'])).resize(1600,1600)
        original=c.GaussianData.load(ORIGINAL);rgb(pan/'original.png',cam,original)
        del original;gc.collect();torch.cuda.empty_cache()
        shutil.copy2(qwen_image(row),pan/'m2_reference.png')
        p=Path(row['ours_source'])/'m3/object_refined_local.ply';asset=c.GaussianData.load(p)
        camera_path=p.parent/'pbr_views/cameras.json';cams=r.read(camera_path)['views']
        rec=next(x for x in cams if x['name']==choices[case])
        ac=c.Camera.from_record(rec).resize(1600,1600)
        xyz=asset.xyz.cpu().numpy();center=(np.quantile(xyz,.005,axis=0)+np.quantile(xyz,.995,axis=0))/2
        # Camera-only framing to consistent 78% maximum silhouette extent.
        direction=ac.center-center;distance=np.linalg.norm(direction);direction/=distance
        for _ in range(3):
            stats=r.alpha_stats(r.object_alpha(ac.resize(320,320),asset));b=stats['bbox']
            fraction=max(b[2]-b[0],b[3]-b[1])/320
            distance*=fraction/.78
            ac=c.Camera(ac.R,-ac.R@(center+direction*distance),1600,1600,ac.fx,ac.fy,ac.cx,ac.cy,ac.name)
        rgb(pan/'m3_gaussian_asset.png',ac,asset,True)
        asset_count=len(asset.xyz);del asset;gc.collect();torch.cuda.empty_cache()
        final=c.GaussianData.load(row['ours_scene']);rgb(pan/'m4_integrated_scene.png',cam,final)
        del final;gc.collect();torch.cuda.empty_cache()
        manifest.append(dict(case=case,A=row['A'],B=row['ours_saved_B'],
            original=source(ORIGINAL),qwen=source(qwen_image(row)),m3_asset=source(p),m3_gaussians=asset_count,
            m3_camera_source=source(camera_path),m3_camera=ac.record(),m3_pbr_steps=4000,
            final_scene=source(row['ours_scene']),world_object=source(final_object(row)),
            scene_camera=cam.record(),anchor_camera=source(row['ours_anchor']),
            m4_local_background_steps=2000,previous_automatic_guard_passed=row.get('ours_guard_passed'),
            same_original_final_camera=True,only_camera_framing_adjusted=True,source_ply_modified=False))
        for name in ['original','m2_reference','m3_gaussian_asset','m4_integrated_scene']:
            images.append((case+' '+name,pan/(name+'.png')))
        print('F06',case,'COMPLETE',flush=True)
    r.save(dest/'case_manifest.json',dict(cases=manifest,render='Actual saved Gaussian PLY, stored spherical harmonics, shared rasterizer.'))
    contact(images,dest/'panels_preview.jpg',columns=4,size=320)


@torch.no_grad()
def f10():
    row=CASES[0];dest=OUT/'F10_multiview';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'figure_F10.pdf').exists(), 'Completed publication figure exists'
    full=dest/'full_scene';detail=dest/'equipment_detail';full.mkdir(exist_ok=True);detail.mkdir(exist_ok=True)
    scene=c.GaussianData.load(row['ours_scene']);obj=c.GaussianData.load(final_object(row))
    original_ref=accepted_cam('case00');region=r.read(Path(row['target_record']).parent/'edit_region.json')
    up=c.normalized(np.array(region['up']));xyz=obj.xyz.cpu().numpy()
    center=(np.quantile(xyz,.005,axis=0)+np.quantile(xyz,.995,axis=0))/2
    distance=np.linalg.norm(original_ref.center-center)*.85
    ref=c.Camera.from_record(original_ref.record());ref.fx*=.85;ref.fy*=.85
    # Front is the accepted anchor direction. Other cameras orbit the same object.
    directions=[('a_front',0,0),('b_left_oblique',-45,0),('c_right_side',90,20),('d_rear',180,20),('e_elevated',0,27)]
    flags=torch.zeros_like(scene.xyz);flags[:,1]=1;flags[-len(obj.xyz):,0]=1
    records=[];boxes=[]
    for name,yaw,pitch in directions:
        cam=r.camera_at(ref,center,center,distance,up,yaw,pitch,name).resize(2048,2048)
        metrics=r.candidate_metrics(cam,obj,scene,flags,256)
        # Only adjust camera distance when the independently rendered object clips.
        factor=1.
        while metrics['capture']<.995 and factor<1.5:
            factor*=1.1
            cam=r.camera_at(ref,center,center,distance*factor,up,yaw,pitch,name).resize(2048,2048)
            metrics=r.candidate_metrics(cam,obj,scene,flags,256)
        rgb(full/(name+'.png'),cam,scene)
        box=r.alpha_stats(r.object_alpha(cam,obj))['bbox'];boxes.append(box)
        records.append(dict(view=name,camera=cam.record(),yaw_offset_degrees=yaw,pitch_offset_degrees=pitch,
            distance=float(distance*factor),object_alpha_bounds=box,actual_visibility=metrics,
            framing='Camera orbit only. True foreground/background occlusion retained.'))
        print('F10',name,'visibility',metrics['visibility'],flush=True)
    side=int(np.ceil(max(max(b[2]-b[0],b[3]-b[1]) for b in boxes)*1.18))
    side=min(side,2048)
    for record,box in zip(records,boxes):
        x=int(round((box[0]+box[2]-side)/2));y=int(round((box[1]+box[3]-side)/2))
        x=max(0,min(x,2048-side));y=max(0,min(y,2048-side));crop=[x,y,x+side,y+side]
        Image.open(full/(record['view']+'.png')).crop(crop).save(detail/(record['view']+'.png'))
        record['detail_crop_xyxy']=crop
    r.save(dest/'cameras.json',dict(reference_camera=original_ref.record(),orbit_center=center,up=up,
        shared_crop_size=side,detail_is_same_camera_crop=True,views=records,
        camera_selection='Shared orbit distance and focal length are 85% of the accepted reference. Side/rear pitch raised 20 degrees to see over real storage bins; elevated view +27 degrees. Scene untouched.'))
    r.save(dest/'sources.json',dict(case='case00',scene=source(row['ours_scene']),object=source(final_object(row)),
        f05_scene_sha256=r.read(OUT/'F05_case00/sources.json')['final_scene']['sha256'],same_scene_as_f05=True,
        source_ply_modified=False,novel_scene_visibility_limitation='Unobserved factory surfaces and foreground occluders are not repaired.'))
    contact([(x['view'],full/(x['view']+'.png')) for x in records]+[(x['view']+' detail',detail/(x['view']+'.png')) for x in records],dest/'panels_preview.jpg',columns=5,size=320)
    print('F10 PANELS COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['F05','F06-assets','F06','F10'],required=True)
    parser.add_argument('--asset-choices',default='view_032,view_034,view_025,view_035,view_026')
    args=parser.parse_args();torch.cuda.set_per_process_memory_fraction(.45)
    if args.figure=='F05':f05()
    elif args.figure=='F06-assets':f06_assets()
    elif args.figure=='F06':f06(dict(zip([x['case'] for x in CASES],args.asset_choices.split(','))))
    else:f10()
