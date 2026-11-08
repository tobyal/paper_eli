"""Read-only M3 placement audit; renders a separately labelled height-only control.

Run with the GSLE interpreter on ubuntuA6000. No image/3D generation or training.
All diagnostic outputs live in a new directory; completed stage files are untouched.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .core.context import Context, gsle, read_json, write_json, sha256
from .core.scene import read_vertices


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    ctx=Context(args.config); out=ctx.path(args.output); out.mkdir(parents=True,exist_ok=False)
    c=gsle(ctx)
    region=read_json(ctx.stage('m1')/'edit_region.json')
    manifest=read_json(ctx.stage('m1')/'manifest.json')
    transform=read_json(ctx.stage('m3')/'transform.json')
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    estimate=region['up_estimation']
    normal=np.asarray(estimate['normal'],dtype=float)
    norm=np.linalg.norm(normal); normal/=norm; offset=estimate['offset']/norm
    local=read_vertices(ctx.stage('m3')/'object_local.ply')
    world=read_vertices(ctx.stage('m3')/'object_world.ply')
    xyz=np.column_stack([world[n] for n in ('x','y','z')]).astype(float)
    local_xyz=np.column_stack([local[n] for n in ('x','y','z')]).astype(float)
    height=xyz@normal+offset
    lift=float(-height.min())
    reconstruction=local_xyz@np.asarray(transform['rotation']).T*transform['scale']+transform['translation']
    base=height<=height.min()+.15*np.ptp(height)
    base_fraction=float(base.mean())
    background=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    original=c.GaussianData.load(ctx.stage('m3')/'object_world.ply')
    moved=c.GaussianData(original.xyz+torch.tensor(lift*normal,device='cuda',dtype=torch.float32),
        original.log_scale,original.rotation,original.opacity_logit,original.sh)
    metrics={}
    with torch.no_grad():
        for name,obj in [('original',original),('height_only_control',moved)]:
            c.save_image(out/(name+'_isolated.png'),c.render(camera,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'])
            c.save_image(out/(name+'_scene.png'),c.render_joint(camera,background,obj,aux=False)['rgb'])
            # Same rasterizer, opacity, scale and depth sorting. Background is
            # black and only the bottom 15% of object height emits white.
            flags=torch.tensor(base,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)
            isolated=c.raster(camera,obj,flags)[0]
            joint=c.concatenate(background,obj)
            visible=c.raster(camera,joint,torch.cat((torch.zeros((len(background.xyz),3),device='cuda'),flags)))[0]
            c.save_image(out/(name+'_base_isolated_alpha.png'),isolated)
            c.save_image(out/(name+'_base_visible_alpha.png'),visible)
            metrics[name]={'base_visibility_ratio':float(visible.sum()/isolated.sum()),
                'base_alpha_mass_isolated':float(isolated.sum()),'base_alpha_mass_in_scene':float(visible.sum())}
            del joint,visible,isolated
    projected,_=camera.project(xyz[base])
    low,high=projected.min(0),projected.max(0)
    report={
        'source_run_config':str(ctx.config_path),'camera':camera.record(),
        'original_scene_sha256':sha256(ctx.stage('m3')/'scene_coarse.ply'),
        'object_world_sha256':sha256(ctx.stage('m3')/'object_world.ply'),
        'plane':{'normal':normal.tolist(),'offset':float(offset),'estimation':estimate['method']},
        'used_floor_local_y':transform['floor_local_y'],
        'estimated_floor_local_y':float(-(np.asarray(manifest['region_center'])@normal+offset)),
        'signed_height_quantiles':np.quantile(height,[0,.01,.05,.1,.5,1]).tolist(),
        'below_floor_count':int((height<0).sum()),'object_count':len(height),
        'below_floor_fraction':float((height<0).mean()),
        'height_only_correction':lift,'original_object_height':float(np.ptp(height)),
        'uniform_transform_max_position_error':float(np.abs(reconstruction-xyz).max()),
        'base_definition':'bottom 15% of world-up object height, without semantic labels',
        'base_gaussian_fraction':base_fraction,'base_visibility':metrics,
        'base_projected_bbox':[low.tolist(),high.tolist()],
        'control_limit':'Only translates along the existing estimated ground normal. No re-fit, new model, background changes or final production replacement.',
    }
    write_json(out/'diagnosis.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
