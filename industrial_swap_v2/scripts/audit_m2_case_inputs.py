"""Read-only alignment, provenance and no-generation audit of prepared M2 inputs."""
import argparse
import sys
import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from ..core.context import Context,read_json,write_json,sha256,gsle


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args();ctx=Context(a.config);c=gsle(ctx)
    out=ctx.stage('m2');conditions=read_json(out/'conditions.json');camera=c.Camera.from_record(conditions['camera'])
    region=read_json(a.case/'outputs/m1/edit_region.json');mask=np.asarray(Image.open(out/'edit_mask.png'))>127
    adaptive=conditions.get('spatial_method')=='adaptive_visible_v1'
    if adaptive:
        from ..core.m2_space import generation_frame
        region=read_json(out/'generation_region.json')
        projected=c.bbox_mask(camera,np.asarray(region['bbox_corners'])).astype(bool)
        np.testing.assert_array_equal(projected,np.asarray(Image.open(out/'generation_bbox_mask.png'))>127)
        assert generation_frame(ctx)==region['world_transform']
        protected=np.asarray(Image.open(out/'protected_foreground_mask.png'))>127
        assert not (mask&protected).any()
        old=np.asarray(Image.open(out/'old_clean_mask.png'))>127
        assert not (mask&~projected&~old).any()
    else:
        np.testing.assert_array_equal(mask,c.bbox_mask(camera,np.asarray(region['bbox_corners'])).astype(bool))
    images={n:np.asarray(Image.open(out/n).convert('RGB')) for n in ('anchor.png','position_hint.png','depth.png','depth_background.png')}
    assert all(im.shape==(camera.height,camera.width,3) for im in images.values())
    assert (images['depth_background.png'][mask]==0).all()
    np.testing.assert_array_equal(images['depth_background.png'][~mask],images['depth.png'][~mask])
    if adaptive:np.testing.assert_array_equal(images['position_hint.png'][~mask],images['anchor.png'][~mask])
    assert mask.any() and (~mask).any()
    raw=np.load(out/'depth.npz');np.testing.assert_array_equal(raw['edit_mask'].astype(bool),mask)
    assert raw['depth'].shape==(1,camera.height,camera.width)
    assert np.isfinite(raw['depth']).all() and np.isfinite(raw['alpha']).all()
    plan=read_json(out/'generation_plan.json');inputs=read_json(out/'generation_inputs.json')
    assert plan['prompt']==(out/'generation_prompt.txt').read_text()
    assert [Path(p).name for p in plan['images']]==inputs['image_order']==['anchor.png','position_hint.png','depth_background.png']
    for image in plan['images']:assert Path(image).parent==out
    assert conditions['anchor_reused'] is False
    reference=read_json(a.case/'reference/camera.json')
    same_pose=np.allclose(reference['R'],camera.R) and np.allclose(reference['t'],camera.t)
    selection=conditions['avp_selection'];observed_pose_verified=None
    if selection.get('candidate_pool','').startswith('calibrated_observed'):
        chosen=selection['selected'];observed=next(cam for cam in c.source_cameras(938) if cam.name==chosen['source_camera'])
        np.testing.assert_allclose(camera.R,observed.R,atol=1e-7)
        np.testing.assert_allclose(camera.t,observed.t,atol=1e-7)
        left,top,right,bottom=chosen['source_crop_xyxy'];factor=camera.width/(right-left)
        assert right-left==bottom-top
        np.testing.assert_allclose([camera.fx,camera.fy,camera.cx,camera.cy],
            [observed.fx*factor,observed.fy*factor,(observed.cx-left)*factor,(observed.cy-top)*factor],rtol=1e-6)
        observed_pose_verified=True
    for n in ('target_gaussians.ply','background_gaussians.ply','edit_region.json','target_indices.npy'):
        assert sha256(a.case/'outputs/m1'/n)==conditions['input_sha256'][n]
    integrity=read_json(a.case/'integrity.json')
    assert all(sha256(a.case/name)==expected for name,expected in integrity.items())
    assert not list(ctx.records.glob('qwen*lock'))
    assert not (out/'api_request.json').exists() and not (out/'edited_scene.png').exists() and not (out/'object_rgba.png').exists()
    assert not (ctx.out/'m3').exists() and not (ctx.out/'m4').exists()
    write_json(a.report,{'case':str(a.case),'output':str(out),
        'spatial_method':conditions.get('spatial_method','legacy_bbox_projection'),
        'mask_matches_case_3d_region_projection':not adaptive,
        'adaptive_mask_geometry_guards_verified':adaptive,
        'images_and_raw_depth_aligned':True,'depth_zero_inside_unchanged_outside':True,'original_m1_integrity_preserved':True,
        'anchor':camera.name,'m1_reference':reference['name'],'same_camera_pose_as_m1':bool(same_pose),
        'anchor_selected_by':conditions['avp_selection'].get('candidate_pool','original_orbit'),
        'calibrated_observed_pose_and_intrinsics_verified':observed_pose_verified,
        'anchor_quality':selection.get('quality_status','passed'),
        'mask_pixels':int(mask.sum()),'prompt_and_image_order_verified':True,'qwen_api_called':False,
        'sam_called':False,'trellis_called':False,'m4_called':False,'result':'passed'})
    print('M2 input audit passed',camera.name,'same pose as M1:',same_pose,flush=True)


if __name__=='__main__':main()
