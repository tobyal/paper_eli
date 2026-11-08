"""Resolve symmetric M3 silhouette yaw with actual Gaussian RGB rendering.

Reuse the generated local asset in a new run. Keep the existing uniform-scale,
floor-anchored placement and generation-volume constraints. This is pose fitting,
not device training or a second TRELLIS call.
"""
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch
import yaml
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256,gsle
from industrial_swap_v2.core.scene import read_vertices,save_vertices
from industrial_swap_v2.core.alignment import transform_vertices
from industrial_swap_v2.core.m2_space import generation_frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--destination',required=True);p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--diverse-starts',action='store_true',help='Fit translation/scale per yaw basin before joint refinement')
    p.add_argument('--reuse-yaw-basins',type=Path,help='Reuse verified prior basin search')
    p.add_argument('--reference-front-x',action='store_true',help='After visual verification, require native preview-frame-0 (+X) face towards Anchor')
    a=p.parse_args();source=Context(a.source_config);root=source.path(a.destination)
    if (root/'records/m3_status.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    cfg=source.cfg;cfg['project'].update(output_dir=a.destination,gpu=a.gpu,
        records_dir=str((root/'records').relative_to(source.base)),logs_dir=str((root/'logs').relative_to(source.base)))
    cfg['m3']['input_image']=str((root/'m2/object_rgba.png').relative_to(source.base))
    cfg['m3']['reference_appearance_alignment']={'resolution':192,'rgb_weight':2.,'yaw_grid_degrees':15,'pose_only':True}
    cfg['m3']['reference_appearance_alignment']['diverse_starts']=a.diverse_starts
    cfg['m3']['reference_appearance_alignment']['verified_native_front_x_prior']=a.reference_front_x
    for stage in ('m1','m2'):
        if not (root/stage).exists():(root/stage).symlink_to(source.stage(stage).resolve(),target_is_directory=True)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    ctx=Context(root/'config.yaml');os.environ.update(ctx.env(gpu=a.gpu));os.environ['SWAP_DISABLE_QWEN']='1'
    out=ctx.stage('m3');old=source.stage('m3')
    for name in ('object_local.ply','object_local.json','trellis_asset.glb','trellis_asset_preview.png','trellis_asset.mp4','trellis_native.pt','trellis_generation.json','trellis_condition.png'):
        if (old/name).exists() and not(out/name).exists():(out/name).symlink_to((old/name).resolve())
    c=gsle(ctx);local=c.GaussianData.load(out/'object_local.ply');vertices=read_vertices(out/'object_local.ply')
    points=np.column_stack([vertices[k] for k in ('x','y','z')]);frame=generation_frame(ctx)
    axes=np.asarray(frame['axes']);center=np.asarray(frame['center']);low=np.asarray(frame['local_bbox_low']);high=np.asarray(frame['local_bbox_high']);extent=high-low
    original=read_json(old/'transform.json');camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json')).resize(192,192)
    camera_record=camera.record();camera_position=-np.asarray(camera_record['R']).T@np.asarray(camera_record['t'])
    rgba=Image.open(ctx.stage('m2')/'object_rgba.png').convert('RGBA').resize((192,192),Image.Resampling.LANCZOS)
    alpha=np.asarray(rgba.getchannel('A'))/255.;white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
    truth=torch.tensor(alpha,device='cuda',dtype=torch.float32);target=torch.tensor(np.asarray(white.convert('RGB')).transpose(2,0,1).copy(),device='cuda',dtype=torch.float32)/255
    colors=(local.sh[:,:,0]*.28209479177387814+.5).clamp_min(0).contiguous();ones=torch.ones_like(colors)
    white_bg=torch.ones(3,device='cuda');trace=[]
    def placement(params):
        yaw,fraction,dx,dz=params;lr=Rotation.from_euler('y',yaw,degrees=True).as_matrix()
        rotated=points@lr.T;lo,hi=rotated.min(0),rotated.max(0)
        scale=float(np.min(extent/np.maximum(hi-lo,1e-8))*fraction)
        offset=(low+high)/2-scale*(lo+hi)/2;offset[1]=low[1]-scale*lo[1]
        offset[[0,2]]+=np.array([dx,dz])*extent[[0,2]]
        rot=axes@lr;shift=center+axes@offset
        excess=np.maximum(low-(rotated*scale+offset),0)+np.maximum(rotated*scale+offset-high,0)
        penalty=5*float(np.mean((excess/extent)**2))
        return scale,rot,shift,penalty
    def data(params):
        scale,rot,shift,penalty=placement(params)
        q=torch.tensor(Rotation.from_matrix(rot).as_quat()[[3,0,1,2]],device='cuda',dtype=torch.float32)
        b=torch.nn.functional.normalize(local.rotation,dim=-1);bw,bv=b[:,:1],b[:,1:];qw,qv=q[:1],q[1:]
        quat=torch.cat((qw*bw-(qv*bv).sum(-1,keepdim=True),qw*bv+bw*qv+torch.cross(qv.expand_as(bv),bv,dim=-1)),dim=-1)
        xyz=local.xyz@torch.tensor(rot.T,device='cuda',dtype=torch.float32)*scale+torch.tensor(shift,device='cuda',dtype=torch.float32)
        return c.GaussianData(xyz,local.log_scale+np.log(scale),quat,local.opacity_logit,local.sh),penalty
    @torch.no_grad()
    def evaluate(params,keep=False):
        obj,penalty=data(params);rgb=c.raster(camera,obj,colors,white_bg);rendered=c.raster(camera,obj,ones)[0].clamp(0,1)
        iou=(torch.minimum(rendered,truth).sum()/torch.maximum(rendered,truth).sum().clamp_min(1)).item()
        error=(((rgb-target).abs()*truth[None]).sum()/(truth.sum().clamp_min(1)*3)).item()
        value=1-iou+2*error+penalty
        _,world_rot,world_shift,_=placement(params)
        direction=camera_position-world_shift;direction/=np.linalg.norm(direction)
        face_cosine=float(world_rot[:,0]@direction)
        if a.reference_front_x and face_cosine<.2:value+=10*(.2-face_cosine)
        record={'parameters':list(map(float,params)),'soft_iou':iou,'foreground_rgb_mae':error,'space_penalty':penalty,'objective':value}
        record['reference_face_cosine']=face_cosine
        if keep:trace.append(record)
        return value,record,rgb
    initial=original['fit_parameters'];initial_value,initial_metrics,initial_rgb=evaluate(initial,True)
    candidates=[(initial_value,initial)]
    for yaw in np.arange(-180,180,15):
        for fraction in (initial[1],cfg['m3']['fit_fraction']):
            values=[float(yaw),fraction,initial[2],initial[3]];score,_,_=evaluate(values,True);candidates.append((score,values))
    candidates.sort(key=lambda x:x[0]);best=candidates[0]
    basin_results=[]
    if a.reuse_yaw_basins:
        prior=read_json(a.reuse_yaw_basins)
        assert prior['source_object_local_sha256']==sha256(out/'object_local.ply')
        assert prior['source_reference_sha256']==sha256(ctx.stage('m2')/'object_rgba.png')
        for record in prior.get('yaw_basin_results',[]):
            score,checked,_=evaluate(record['parameters']);basin_results.append(checked);candidates.append((score,checked['parameters']))
    elif a.diverse_starts:
        for yaw in np.arange(-180,180,15):
            start=min((x for x in candidates if abs(x[1][0]-yaw)<1e-5),key=lambda x:x[0])[1]
            fitted=minimize(lambda x:evaluate([yaw,*x])[0],start[1:],method='Powell',
                bounds=[(.25,1),(-.35,.35),(-.35,.35)],options={'maxiter':12,'xtol':.002,'ftol':.0002})
            params=[float(yaw),*fitted.x.tolist()];score,record,_=evaluate(params);basin_results.append(record)
            candidates.append((score,params))
    if a.diverse_starts or a.reuse_yaw_basins:
        if a.reference_front_x:candidates=[x for x in candidates if evaluate(x[1])[1]['reference_face_cosine']>=.2]
        candidates.sort(key=lambda x:x[0]);best=candidates[0]
        starts=[]
        for candidate in candidates:
            if all(abs((candidate[1][0]-old[1][0]+180)%360-180)>=30 for old in starts): starts.append(candidate)
            if len(starts)==3:break
    else:starts=candidates[:3]
    for _,start in starts:
        yaw_bounds=(max(-180,start[0]-20),min(180,start[0]+20)) if a.diverse_starts or a.reuse_yaw_basins else (-180,180)
        fit=minimize(lambda x:evaluate(x)[0],start,method='Powell',bounds=[yaw_bounds,(.25,1),(-.35,.35),(-.35,.35)],options={'maxiter':20,'xtol':.002,'ftol':.0002})
        if fit.fun<best[0]:best=(float(fit.fun),fit.x.tolist())
    _,metrics,rgb=evaluate(best[1]);scale,rotation,translation,_=placement(best[1])
    save_vertices(out/'object_world.ply',transform_vertices(vertices,scale,rotation,translation))
    matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=translation
    write_json(out/'transform.json',{**original,'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),
        'object_to_world':matrix.tolist(),'world_to_object':np.linalg.inv(matrix).tolist(),'fit_parameters':best[1],
        'silhouette_objective':1-metrics['soft_iou'],'reference_rgb_mae':metrics['foreground_rgb_mae'],
        'method':'uniform scale, floor-anchored placement, actual Gaussian silhouette and reference RGB pose fitting',
        'generation_frame':frame,'original_transform':str(old/'transform.json')})
    write_json(out/'reference_alignment.json',{'source_m3':str(old),'source_object_local_sha256':sha256(old/'object_local.ply'),
        'source_object_world_sha256':sha256(old/'object_world.ply'),'source_reference_sha256':sha256(ctx.stage('m2')/'object_rgba.png'),
        'initial':initial_metrics,'selected':metrics,'coarse_candidates':trace,'yaw_basin_results':basin_results,'trellis_called':False,
        'gaussian_count_unchanged':len(vertices),'device_shape_texture_opacity_unchanged':True,
        'volume':frame,'verified_native_front_x_prior':a.reference_front_x,
        'loss':'1 - soft silhouette IoU + 2 * reference foreground RGB MAE + relative volume penalty; optional verified reference-face orientation constraint'})
    def pil(t):return Image.fromarray(np.rint(t.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype('uint8'))
    board=Image.new('RGB',(960,354),'white');draw=ImageDraw.Draw(board)
    for j,(im,label) in enumerate([(white.convert('RGB'),'Reference'),(pil(initial_rgb),'Silhouette-only pose'),(pil(rgb),'RGB + silhouette pose')]):
        board.paste(im.resize((320,320)),(j*320,34));draw.text((j*320+8,9),label,fill='black')
    board.save(out/'alignment_comparison.jpg')
    from industrial_swap_v2.core.m3_composition import compose_m3
    from industrial_swap_v2.core.rendering import render_outputs
    background=compose_m3(ctx);render_outputs(ctx,'m3',out/'object_world.ply',background_path=background)
    write_json(out/'reuse.json',{'trellis_called':False,'source':str(old),'adaptation':'reference appearance disambiguation after silhouette fitting'})
    ctx.complete('m3',[out/n for n in ('object_world.ply','scene_coarse.ply','transform.json','reference_alignment.json','renders/anchor_after.png')])
    print('REFERENCE ALIGNMENT',initial_metrics,metrics,flush=True)


if __name__=='__main__':main()
