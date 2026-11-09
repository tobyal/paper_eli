"""Preview and save a user-requested rigid yaw correction, with no training."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.context import Context, gsle, read_json, write_json, sha256
from industrial_swap_v2.core.scene import read_vertices, save_vertices, compose
from industrial_swap_v2.core.alignment import transform_vertices
from industrial_swap_v2.core.gaussian_registration import Pose, DEFAULTS, reference, image_metrics, numbers
from industrial_swap_v2.core.m2_space import registration_space


def board(paths, labels, output, width=600, columns=3):
    canvas=Image.new('RGB',(width*columns,(width+32)*math.ceil(len(paths)/columns)),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(path,label) in enumerate(zip(paths,labels)):
        x=i%columns*width;y=i//columns*(width+32)
        image=ImageOps.contain(Image.open(path).convert('RGB'),(width,width))
        canvas.paste(image,(x,y+32));draw.text((x+8,y+8),label,fill='black')
    canvas.save(output,quality=94)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-image')
    parser.add_argument('--delta-yaw',type=float)
    args=parser.parse_args();ctx=Context(args.source_config);c=gsle(ctx)
    out=ctx.path(args.output);out.mkdir(parents=True,exist_ok=True)
    original=ctx.stage('m4');transform=read_json(original/'transform.json')
    summary=read_json(original/'summary.json')
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    reference_image=Path(args.reference_image) if args.reference_image else ctx.stage('m2')/'edited_scene.png'
    cfg={**DEFAULTS,**ctx.cfg['m4']['integration'].get('registration',{})}
    local=c.GaussianData.load(ctx.stage('m3')/'object_refined_local.ply')
    model=Pose(c,local,registration_space(ctx),cfg)
    background=c.GaussianData.load(summary['candidate_background'])
    p0=model.tensor(transform['parameters_radians_logscale_dx_dz'])
    cam=camera.resize(640,round(camera.height*640/camera.width))
    ref=reference(ctx,camera,640,local.xyz.device)

    @torch.no_grad()
    def candidate(delta,name):
        p=p0.clone();p[0]+=math.radians(delta)
        data=model.data(p);scale,rotation,translation,_,_=model.placement(p)
        center=(model.tensor(cam.center)-translation)@rotation/scale
        joint=c.GaussianData(*(torch.cat((getattr(background,k),getattr(data,k)))
            for k in ('xyz','log_scale','rotation','opacity_logit')),sh=None)
        colors=torch.cat((background.colors(cam.center),local.colors(center)))
        c.save_image(out/(name+'.png'),c.raster(cam,joint,colors))
        pred=model.render(p,cam);c.save_image(out/(name+'_object.png'),pred['rgb'])
        return numbers(image_metrics(pred,ref))

    if args.delta_yaw is None:
        rows=[];paths=[reference_image];labels=['User reference']
        for delta in (-15,-10,-5,0,5,10,15):
            name=f'yaw_{delta:+03d}';metrics=candidate(delta,name)
            rows.append({'delta_yaw':delta,**metrics});paths.append(out/(name+'.png'))
            labels.append(f'Yaw {delta:+d} deg | IoU {metrics["mask_iou"]:.3f}')
        board(paths,labels,out/'yaw_candidates.jpg',width=480,columns=4)
        write_json(out/'yaw_candidates.json',rows)
        print(out/'yaw_candidates.jpg',flush=True);return

    if (out/'object_world.ply').exists():raise ValueError('Final correction exists; choose a new output directory')
    candidate(0,'before');metrics=candidate(args.delta_yaw,'after')
    sources=[original/'object_world.ply',ctx.stage('m3')/'object_refined_local.ply',
        Path(summary['candidate_background']),Path(summary['guarded_background']),original/'transform.json']
    hashes={str(path):sha256(path) for path in sources}
    # Rotate the already exported object exactly around its ground support axis.
    # Geometry, covariance and all SH bands undergo the same rigid transform.
    axis=np.asarray(transform['generation_frame']['axes'])[:,1]
    axis/=np.linalg.norm(axis);pivot=np.asarray(transform['support_center_world'])
    delta=Rotation.from_rotvec(axis*math.radians(args.delta_yaw)).as_matrix()
    vertices=read_vertices(original/'object_world.ply')
    rotated=transform_vertices(vertices,1.,delta,pivot-delta@pivot)
    assert len(rotated)==len(vertices)
    for field in ('opacity','scale_0','scale_1','scale_2'):
        assert rotated[field].tobytes()==vertices[field].tobytes()
    save_vertices(out/'object_world.ply',rotated)
    new_rotation=delta@np.asarray(transform['rotation'])
    new_translation=pivot+delta@(np.asarray(transform['translation'])-pivot)
    matrix=np.eye(4);matrix[:3,:3]=transform['scale']*new_rotation;matrix[:3,3]=new_translation
    adjusted={k:transform[k] for k in ('scale','camera','generation_frame','floor_local_y','source_local','source_local_sha256') if k in transform}
    adjusted.update(method='user_requested_rigid_yaw_correction',delta_yaw_degrees=args.delta_yaw,
        yaw_degrees=(transform['yaw_degrees']+args.delta_yaw)%360,rotation=new_rotation.tolist(),
        translation=new_translation.tolist(),object_to_world=matrix.tolist(),world_to_object=np.linalg.inv(matrix).tolist(),
        support_center_world=pivot.tolist(),source_transform=str(original/'transform.json'))
    write_json(out/'transform.json',adjusted)
    del model,local,background;torch.cuda.empty_cache()
    compose(summary['candidate_background'],out/'object_world.ply',out/'scene_rotated_step2000.ply',out/'composition_step2000.json')
    compose(summary['guarded_background'],out/'object_world.ply',out/'scene_rotated_guarded.ply',out/'composition_guarded.json')
    obj=c.GaussianData.load(out/'object_world.ply');bg=c.GaussianData.load(summary['candidate_background'])
    old=c.GaussianData.load(original/'object_world.ply')
    with torch.no_grad():
        c.save_image(out/'anchor_after.png',c.render_joint(camera,bg,obj,aux=False)['rgb'])
        c.save_image(out/'anchor_before.png',c.render_joint(camera,bg,old,aux=False)['rgb'])
        c.save_image(out/'object_anchor.png',c.render(camera,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'])
        manifest=read_json(original/'multiview_supervision/manifest.json')
        views=[v for v in manifest['views'] if v['split']=='validation']
        for i in (len(views)//3,2*len(views)//3):
            rec=views[i];cam2=c.Camera.from_record(rec['camera'])
            c.save_image(out/(rec['name']+'_before.png'),c.render_joint(cam2,bg,old,aux=False)['rgb'])
            c.save_image(out/(rec['name']+'_after.png'),c.render_joint(cam2,bg,obj,aux=False)['rgb'])
            board([out/(rec['name']+'_before.png'),out/(rec['name']+'_after.png')],
                ['Before','Yaw corrected'],out/(rec['name']+'_comparison.jpg'),columns=2)
    board([reference_image,out/'anchor_before.png',out/'anchor_after.png'],
        ['User reference','Before',f'Yaw corrected {args.delta_yaw:+g} deg'],out/'comparison.jpg')
    assert all(sha256(path)==digest for path,digest in hashes.items())
    write_json(out/'report.json',{'source_config':str(ctx.config_path),'source_hashes':hashes,
        'reference_image':str(reference_image),'reference_sha256':sha256(reference_image),
        'delta_yaw_degrees':args.delta_yaw,'anchor_metrics':metrics,'source_gaussians':len(vertices),
        'scale_unchanged':True,'support_center_unchanged':True,'opacity_unchanged':True,
        'background_rows_unchanged':True,'source_files_unchanged':True,
        'background_training_steps_reused':summary['checks']['training_steps'],'new_training_steps':0,
        'background_source_guard_passed':summary['candidate_guard_passed'],
        'note':'Rigid pose correction only. Existing 2000-step and guarded backgrounds reused separately; no new M4 optimization or guard evaluation.'})
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
