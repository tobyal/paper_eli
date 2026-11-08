"""Verify immutable inputs and export genuine 3D repair comparisons/videos."""
import argparse
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from .core.context import Context, gsle, read_json, write_json, sha256
from .core.scene import read_vertices
from .core.rendering import VideoWriter


def uint8(tensor):
    return np.round(tensor.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--repair',required=True)
    args=p.parse_args(); ctx=Context(args.config); dest=ctx.path(args.repair)
    out=dest/'validation'; out.mkdir(exist_ok=False)
    report=read_json(dest/'repair_report.json')
    inputs={'background':ctx.stage('m1')/'background_gaussians.ply',
            'object':ctx.stage('m3')/'object_world.ply',
            'transform':ctx.stage('m3')/'transform.json',
            'm1_manifest':ctx.stage('m1')/'manifest.json'}
    hashes={k:sha256(v) for k,v in inputs.items()}
    assert hashes==report['source_hashes'], 'An immutable source changed'
    original=read_vertices(inputs['background']); obj_vertices=read_vertices(inputs['object'])
    fixed=read_vertices(dest/'background_fixed.ply'); complete=read_vertices(dest/'scene_coarse_fixed.ply')
    keep=np.load(dest/'retained_background_indices.npy'); removed=np.load(dest/'removed_background_indices.npy')
    yellow=np.load(dest/'protected_yellow_indices.npy')
    assert np.intersect1d(removed,yellow).size==0
    assert np.array_equal(np.sort(np.concatenate((keep,removed))),np.arange(len(original)))
    assert fixed[:len(keep)].tobytes()==original[keep].tobytes()
    assert complete[:len(fixed)].tobytes()==fixed.tobytes()
    assert complete[len(fixed):].tobytes()==obj_vertices.tobytes()
    assert np.isfinite(complete.view(np.float32).reshape(len(complete),-1)).all()
    manifest=read_json(inputs['m1_manifest'])
    axes0=np.asarray(manifest['world_transform']['axes']); center0=np.asarray(manifest['region_center'])
    background_local=(np.column_stack([original[n] for n in ('x','y','z')])-center0)@axes0
    protect=np.zeros(len(original),bool); protect[yellow]=True
    protect&=(np.abs(background_local[:,0])<.8)&(np.abs(background_local[:,2])<1)&(np.abs(background_local[:,1]-report['local_plane_y_ax_bz_c'][2])<.15)
    kept_flags=protect[keep]
    yellow_before=np.r_[protect,np.zeros(len(obj_vertices),bool)]
    yellow_after=np.r_[kept_flags,np.zeros(len(fixed)-len(keep)+len(obj_vertices),bool)]
    del original,complete,fixed
    c=gsle(ctx); camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    bg=c.GaussianData.load(inputs['background']); repaired=c.GaussianData.load(dest/'background_fixed.ply')
    obj=c.GaussianData.load(inputs['object']); m1=read_json(inputs['m1_manifest'])
    axes=np.asarray(m1['world_transform']['axes']); center=np.asarray(m1['region_center']); up=np.asarray(m1['world_up'])
    local=(obj.xyz.cpu().numpy()-center)@axes
    base=local[:,1]<local[:,1].min()+.15*np.ptp(local[:,1])
    flags=torch.tensor(base,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)
    before_joint=c.concatenate(bg,obj); after_joint=c.concatenate(repaired,obj)
    before_flags=torch.cat((torch.zeros((len(bg.xyz),3),device='cuda'),flags))
    after_flags=torch.cat((torch.zeros((len(repaired.xyz),3),device='cuda'),flags))
    yellow_before_flags=torch.tensor(yellow_before,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)
    yellow_after_flags=torch.tensor(yellow_after,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)
    direction=camera.center-center; radius=np.linalg.norm(direction)
    horizontal=c.normalized(direction-up*np.dot(direction,up)); right=c.normalized(np.cross(up,horizontal))
    elevation=np.arcsin(np.dot(direction/radius,up))
    cameras=c.source_cameras(938)
    selected=[v for v in cameras if v.name in ('frame_000131.jpg','frame_000141.jpg','frame_000151.jpg','frame_000161.jpg','frame_000171.jpg')]
    orbits=[]
    for degrees in (-30,-15,15,30,90,180,270):
        azimuth=math.radians(degrees)
        position=center+radius*(math.cos(elevation)*(math.cos(azimuth)*horizontal+math.sin(azimuth)*right)+math.sin(elevation)*up)
        orbits.append(c.look_at(position,center,up,768,name=f'orbit_{degrees:+04d}'))
    metrics=[]
    with torch.no_grad():
        # Load the delivered composite PLY independently, rather than only
        # validating an in-memory concatenation used by the previews.
        delivered=c.GaussianData.load(dest/'scene_coarse_fixed.ply')
        direct=c.render(camera,delivered,aux=False)['rgb']
        joint=c.render_joint(camera,repaired,obj,aux=False)['rgb']
        delivery_error=float((direct-joint).abs().max())
        assert delivery_error<2e-5
        del delivered
        for cam in [camera,*selected,*orbits]:
            name=Path(cam.name).stem
            before=c.render_joint(cam,bg,obj,aux=False)['rgb']; after=c.render_joint(cam,repaired,obj,aux=False)['rgb']
            c.save_image(out/(name+'_before.png'),before); c.save_image(out/(name+'_after.png'),after)
            c.save_image(out/(name+'_comparison.png'),torch.cat((before,after),dim=2))
            isolated=c.raster(cam,obj,flags).sum().item()
            visible_before=c.raster(cam,before_joint,before_flags).sum().item()
            visible_after=c.raster(cam,after_joint,after_flags).sum().item()
            yellow_mass_before=c.raster(cam,before_joint,yellow_before_flags).sum().item()
            yellow_mass_after=c.raster(cam,after_joint,yellow_after_flags).sum().item()
            diff=(after-before).abs()
            metrics.append({'camera':cam.name,'camera_record':cam.record(),
                'base_visible_before':visible_before/max(isolated,1e-8),'base_visible_after':visible_after/max(isolated,1e-8),
                'base_isolated_pixel_mass':isolated/3,
                'nearby_yellow_visible_mass_before':yellow_mass_before/3,
                'nearby_yellow_visible_mass_after':yellow_mass_after/3,
                'changed_pixel_fraction_gt_2_of_255':float((diff.max(0).values>2/255).float().mean()),
                'whole_image_mean_absolute_change':float(diff.mean())})
        # Continuous orbit through the well-observed side, using the same
        # camera matrix for both panels at each frame.
        arc=[]
        with VideoWriter(out/'orbit_before_after.mp4',18) as writer:
            for degrees in np.r_[np.linspace(-35,35,72),np.linspace(35,-35,72)]:
                azimuth=math.radians(degrees)
                position=center+radius*(math.cos(elevation)*(math.cos(azimuth)*horizontal+math.sin(azimuth)*right)+math.sin(elevation)*up)
                cam=c.look_at(position,center,up,640,name=f'arc_{len(arc):03d}')
                before=c.render_joint(cam,bg,obj,aux=False)['rgb']; after=c.render_joint(cam,repaired,obj,aux=False)['rgb']
                writer.append_data(uint8(torch.cat((before,after),dim=2))); arc.append(cam.record())
        trajectory=cameras[::2]
        with VideoWriter(out/'trajectory_before_after.mp4',12) as writer:
            for cam in trajectory:
                writer.append_data(uint8(torch.cat((c.render_joint(cam,bg,obj,aux=False)['rgb'],
                    c.render_joint(cam,repaired,obj,aux=False)['rgb']),dim=2)))
    validation={'status':'verified_and_exported','original_inputs_unchanged':hashes,
        'object_bitwise_unchanged':True,'retained_background_bitwise_unchanged':True,
        'yellow_gaussians_removed':0,'removed_count':len(removed),'original_background_count':report['original_background_count'],
        'removed_fraction':len(removed)/report['original_background_count'],
        'delivered_ply_vs_joint_render_max_error':delivery_error,'view_metrics':metrics,
        'video_panels':['original_M3','background_repaired_same_object'],
        'orbit_frames':len(arc),'trajectory_frames':len(trajectory),
        'limitations':['Hidden floor is completed from surrounding observed floor; its original texture is not recoverable.',
            'Fixed object support is below the neighboring fitted plane; the new local floor meets its unchanged base and transitions to the surrounding floor.',
            'Original reconstruction artifacts outside the repair area are retained.'],
        'artifacts':{name:{'bytes':(dest/name).stat().st_size,'sha256':sha256(dest/name)}
            for name in ('scene_coarse_fixed.ply','background_fixed.ply','floor_patch.ply','removed_gaussians.ply')}}
    write_json(out/'validation.json',validation)
    write_json(out/'video_cameras.json',{'orbit':arc,'trajectory':[cam.record() for cam in trajectory]})
    source_dir=dest/'code'; source_dir.mkdir(exist_ok=True)
    for name in ('inspect_background_repair.py','repair_background.py','export_background_repair.py'):
        shutil.copyfile(Path(__file__).parent/name,source_dir/name)
    shutil.copyfile(ctx.config_path,dest/'source_config.yaml')
    print('Verified immutable robot, retained background, delivered scene, and exported',len(metrics),'views',flush=True)


if __name__=='__main__': main()
