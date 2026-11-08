"""Independent PLY verification and held-out camera/image collateral audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from ..core.context import Context, gsle, read_json, write_json, sha256
from ..core.scene import read_vertices
from ..core.rendering import VideoWriter


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--filter-output',required=True)
    args=p.parse_args(); ctx=Context(args.config); folder=ctx.path(args.filter_output)
    dest=folder/'validation'; dest.mkdir(exist_ok=False); report=read_json(folder/'occlusion_report.json'); cfg=report['config']
    original=read_vertices(report['input_paths']['background']); objv=read_vertices(report['input_paths']['object'])
    filtered=read_vertices(folder/'background_filtered.ply'); scene=read_vertices(folder/'scene_filtered.ply')
    removed=np.load(folder/'removed_background_indices.npy'); keep=np.load(folder/'retained_background_indices.npy')
    assert np.array_equal(np.sort(np.r_[keep,removed]),np.arange(len(original)))
    assert filtered.tobytes()==original[keep].tobytes()
    assert scene[:len(filtered)].tobytes()==filtered.tobytes() and scene[len(filtered):].tobytes()==objv.tobytes()
    assert len(scene)==len(original)-len(removed)+len(objv)
    assert not (folder/'floor_patch.ply').exists()
    for k in ('background','object','reference_mask'):
        assert sha256(report['input_paths'][k])==report['input_sha256'][k]
    c=gsle(ctx); original_data=c.GaussianData.load(report['input_paths']['background']); obj=c.GaussianData.load(report['input_paths']['object'])
    delivered=c.GaussianData.load(folder/'scene_filtered.ply'); repaired=c.GaussianData.load(folder/'background_filtered.ply')
    anchor=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    voting={m['camera'] for m in report['view_metrics']}
    xyz=obj.xyz.cpu().numpy(); probe=xyz[::max(1,len(xyz)//1000)]
    source=c.source_cameras(938); holdouts=[]
    for cam in source:
        if cam.name in voting: continue
        uv,z=cam.project(probe)
        visible=(z>0)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
        if visible.mean()>.95 and np.ptp(uv[visible],axis=0).prod()/(cam.width*cam.height)>.01:
            holdouts.append(cam)
    if len(holdouts)>12: holdouts=[holdouts[i] for i in np.linspace(0,len(holdouts)-1,12).astype(int)]
    before_joint=c.concatenate(original_data,obj); after_joint=c.concatenate(repaired,obj)
    flags_before=torch.cat((torch.zeros((len(original),3),device='cuda'),torch.ones((len(objv),3),device='cuda')))
    flags_after=torch.cat((torch.zeros((len(filtered),3),device='cuda'),torch.ones((len(objv),3),device='cuda')))
    metrics=[]
    def rgb8(t): return np.round(t.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    with torch.no_grad():
        delivered_error=float((c.render(anchor,delivered,aux=False)['rgb']-c.render_joint(anchor,repaired,obj,aux=False)['rgb']).abs().max())
        assert delivered_error<2e-5
        for i,cam in enumerate([anchor,*holdouts]):
            isolated_alpha=c.raster(cam,obj,torch.ones((len(objv),3),device='cuda'))[0]
            mask=isolated_alpha.cpu().numpy()>cfg['mask_alpha_threshold']
            radius=max(1,round(cam.width*.003)); guard=cv2.dilate(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
            before=c.render_joint(cam,original_data,obj,aux=False)['rgb']; after=c.render_joint(cam,repaired,obj,aux=False)['rgb']
            visible_before=c.raster(cam,before_joint,flags_before)[0]; visible_after=c.raster(cam,after_joint,flags_after)[0]
            mass=float(isolated_alpha[mask].sum()); diff=(after-before).abs().cpu().numpy(); changed=diff.max(0)>cfg['outside_change_threshold']
            record={'camera':cam.name,'held_out':cam.name not in voting,'camera_record':cam.record(),
                'visible_before':float(visible_before[mask].sum())/mass,'visible_after':float(visible_after[mask].sum())/mass,
                'outside_mask_mae':float(diff[:,~mask].mean()),'outside_mask_changed_fraction':float(changed[~mask].mean()),
                'outside_guard_mae':float(diff[:,~guard].mean()),'outside_guard_changed_fraction':float(changed[~guard].mean())}
            record['guard_pass']=record['outside_guard_mae']<=cfg['max_outside_mae'] and record['outside_guard_changed_fraction']<=cfg['max_outside_changed_fraction']
            metrics.append(record)
            name=f'{i:02d}_{Path(cam.name).stem}'
            c.save_image(dest/(name+'_before.png'),before); c.save_image(dest/(name+'_after.png'),after)
            c.save_image(dest/(name+'_comparison.png'),torch.cat((before,after),dim=2))
            c.save_image(dest/(name+'_outside_change.png'),diff.max(0)*(~mask))
        # Only actual calibrated cameras. Same object/camera for both panels.
        with VideoWriter(dest/'trajectory_before_after.mp4',12) as writer:
            for cam in source[::2]:
                writer.append_data(rgb8(torch.cat((c.render_joint(cam,original_data,obj,aux=False)['rgb'],
                    c.render_joint(cam,repaired,obj,aux=False)['rgb']),dim=2)))
    result={'status':'passed' if all(m['guard_pass'] for m in metrics) else 'outside_guard_failure',
        'background_points_before':len(original),'background_points_after':len(filtered),'object_points':len(objv),
        'new_gaussians':0,'object_bitwise_unchanged':True,'retained_background_bitwise_unchanged':True,
        'delivered_ply_render_max_error':delivered_error,'outside_change_threshold':cfg['outside_change_threshold'],
        'views':metrics,'video_frames':len(source[::2]),'source_sha256':report['input_sha256']}
    write_json(dest/'validation.json',result)
    print(result['status'],len(metrics),'views;',len(holdouts),'held out',flush=True)
    if result['status']!='passed': raise RuntimeError('Held-out collateral guard failed; retain this candidate for diagnosis only')


if __name__=='__main__': main()
