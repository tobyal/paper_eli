"""Measured common-camera validation; all panels are real 3D Gaussian renders."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ..core.context import gsle, sha256, read_json, write_json
from ..core.scene import read_vertices
from ..core.visibility_loss import object_contribution
from ..core.rendering import VideoWriter


def pixels(t):
    a=t.detach().cpu().numpy() if torch.is_tensor(t) else t
    if a.ndim==3 and a.shape[0]==3: a=a.transpose(1,2,0)
    return np.round(a.clip(0,1)*255).astype(np.uint8)


def board(images,labels,zoom=None):
    arrays=[pixels(a) for a in images]; h,w=arrays[0].shape[:2]
    header=max(34,round(w*.045))
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(14,round(w*.022)))
    except OSError: font=ImageFont.load_default()
    canvas=Image.new('RGB',(w*len(arrays),h+header+(h//2+30 if zoom else 0)),'white'); d=ImageDraw.Draw(canvas)
    for index,(array,label) in enumerate(zip(arrays,labels)):
        image=Image.fromarray(array); canvas.paste(image,(index*w,header)); d.text((index*w+12,10),label,fill='black',font=font)
        if zoom:
            crop=image.crop(zoom); scale=min(w/crop.width,(h//2)/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
            canvas.paste(crop,(index*w+(w-crop.width)//2,h+header+30))
    return canvas


def preview_masks(ctx):
    folder=ctx.stage('m3')/'multiview_visibility'; out=ctx.out/'validation'; out.mkdir(exist_ok=True)
    views=read_json(folder/'manifest.json')['views']; pages=[]
    for record in [views[i] for i in (1,13,25,37)]:
        cache=np.load(folder/'cache'/f"{record['name']}.npz")
        image=cache['original_rgb']/255
        excluded=image.copy(); excluded[cache['unknown_region']]=excluded[cache['unknown_region']]*.4+np.array([.6,0,0])
        trusted=image.copy(); trusted[cache['background_train_mask']]=trusted[cache['background_train_mask']]*.5+np.array([0,.5,0])
        legal=image.copy(); legal[cache['legal_foreground']]=legal[cache['legal_foreground']]*.3+np.array([0,0,.7])
        pages.append(board([image,excluded,trusted,legal],[record['name'],'RED: excluded old/unknown','GREEN: real-photo supervision','BLUE: preserved foreground']))
    panel=Image.new('RGB',(pages[0].width,sum(p.height for p in pages)), 'white'); y=0
    for page in pages: panel.paste(page,(0,y)); y+=page.height
    panel.save(out/'supervision_masks.png')


@torch.no_grad()
def preview_background(ctx):
    c=gsle(ctx); out=ctx.stage('m4')/'background_optimization/renders'; out.mkdir(exist_ok=True)
    bg0=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    bg1=c.GaussianData.load(out.parent/'background_optimized.ply'); obj=c.GaussianData.load(ctx.stage('m3')/'object_world.ply')
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json')).resize(1024,1024)
    images=[c.render_joint(camera,bg,obj,aux=False)['rgb'] for bg in (bg0,bg1)]
    if not ctx.stage('m3').is_symlink():
        c.save_image(ctx.stage('m3')/'renders/anchor_after.png',images[0])
    alpha=c.raster(camera,obj,torch.ones((len(obj.xyz),3),device='cuda'))[0].cpu().numpy()
    y,x=np.where(alpha>.25); zoom=(max(0,int(x.min())-30),max(0,int(y.min())-30),min(1024,int(x.max())+30),min(1024,int(y.max())+30))
    board(images,['M3: original composition','M4 A: optimized background'],zoom).save(out/'anchor_before_after.png')


@torch.no_grad()
def validate(ctx):
    c=gsle(ctx); out=ctx.out/'validation'; out.mkdir(exist_ok=True)
    viewsdir=out/'multiview_comparison'; viewsdir.mkdir(exist_ok=True)
    folder=ctx.stage('m3')/'multiview_visibility'; manifest=read_json(folder/'manifest.json')
    stagea=ctx.stage('m4')/'background_optimization'
    a=read_json(stagea/'summary.json'); b=read_json(ctx.stage('m4')/'summary.json')
    expected_a={v['name']:v for v in read_json(stagea/'final_metrics.json')['views']}
    original_path=ctx.stage('m1')/'background_gaussians.ply'; optimized_path=stagea/'background_optimized.ply'
    object_path=ctx.stage('m3')/'object_world.ply'; refined_path=ctx.stage('m4')/'object_refined.ply'
    assert b['background_sha256']==sha256(optimized_path)==a['background_sha256']
    before=read_vertices(original_path); after=read_vertices(optimized_path); ids=np.load(stagea/'optimized_indices.npy')
    frozen=np.ones(len(before),bool); frozen[ids]=False
    assert len(before)==len(after) and before[frozen].tobytes()==after[frozen].tobytes()
    object_before=read_vertices(object_path); object_after=read_vertices(refined_path)
    geom=['x','y','z']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    for name in geom:
        assert object_before[name].tobytes()==object_after[name].tobytes()
        assert before[name].tobytes()==after[name].tobytes()
    final_vertices=read_vertices(ctx.stage('m4')/'scene_final.ply')
    assert final_vertices[:len(after)].tobytes()==after.tobytes()
    assert final_vertices[len(after):].tobytes()==object_after.astype(after.dtype).tobytes()
    source_preservation=read_json(ctx.out/'source_preservation.json')
    for path,digest in source_preservation.items(): assert sha256(path)==digest,path
    old_proof=ctx.path('outputs/oagof/current_m3_v2/preservation.json')
    historical_checked=0
    if old_proof.exists():
        saved=read_json(old_proof)
        # Historical audit may wrap the path/digest mapping.
        candidates=[saved]+[v for v in saved.values() if isinstance(v,dict)]
        for mapping in candidates:
            for path,digest in mapping.items():
                if isinstance(digest,str) and len(digest)==64 and Path(path).is_file():
                    assert sha256(path)==digest,path
                    historical_checked+=1
    write_json(out/'background_preservation.json',{'frozen_background_rows_bitwise_identical':True,
        'background_count_before':len(before),'background_count_after':len(after),'optimized_rows':len(ids),
        'background_geometry_bitwise_identical':True,'object_geometry_bitwise_identical':True,
        'object_count_before':len(object_before),'object_count_after':len(object_after),
        'original_inputs_unchanged':source_preservation,'historical_outputs_verified':historical_checked,
        'stage_b_uses_stage_a_background':True,'final_ply_components_bitwise_identical':True,
        'deleted_gaussians':0,'added_gaussians':0,'floor_completion':False})
    del before,after,object_before,object_after,final_vertices
    bg0=c.GaussianData.load(original_path); bg1=c.GaussianData.load(optimized_path)
    obj0=c.GaussianData.load(object_path); obj1=c.GaussianData.load(refined_path)
    standalone=c.GaussianData.load(ctx.stage('m4')/'scene_final.ply')
    region=read_json(ctx.stage('m1')/'manifest.json'); up=torch.tensor(region['world_up'],device='cuda')
    height=obj0.xyz@up; base=height<torch.quantile(height,.2)
    base_colors=base[:,None].float().expand(-1,3).contiguous()
    rows=[]; pages=[]; export=[v for v in manifest['views'] if v['split'] in ('anchor','validation')]
    for record in export:
        camera=c.Camera.from_record(record['camera']); name=record['name']; cache=np.load(folder/'cache'/f'{name}.npz')
        colors=[c.render_joint(camera,bg0,obj0,aux=False)['rgb'],c.render_joint(camera,bg1,obj0,aux=False)['rgb'],c.render_joint(camera,bg1,obj1,aux=False)['rgb']]
        independent=c.render(camera,standalone,aux=False)['rgb']
        standalone_error=float((independent-colors[-1]).abs().max())
        assert standalone_error<2e-4,(name,standalone_error)
        mask=cache['object_alpha']>.25; yy,xx=np.where(mask)
        pad=round(min(camera.width,camera.height)*.04)
        zoom=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(camera.width,int(xx.max())+pad),min(camera.height,int(yy.max())+pad))
        image=board(colors,['M3: original composition','M4 A: local background optimization','M4 B: DI-SDS appearance'],zoom)
        image.save(viewsdir/f'{name}.png')
        c.save_image(stagea/'renders'/f'{name}_joint.png',colors[1])
        if record['split']=='anchor': image.save(out/'same_view_comparison.png')
        if len(pages)<4: pages.append(image)
        vrow={'name':name,'split':record['split'],'standalone_render_max_error':standalone_error,'stages':[]}
        reference_base=c.raster(camera,obj0,base_colors)[0]
        for index,(bg,obj) in enumerate([(bg0,obj0),(bg1,obj0),(bg1,obj1)]):
            alpha=c.raster(camera,obj,torch.ones((len(obj.xyz),3),device='cuda'))[0]
            visible=object_contribution(c,camera,bg,obj)[0]
            base_alone=c.raster(camera,obj,base_colors)[0]
            joint=c.concatenate(bg,obj)
            base_joint=c.raster(camera,joint,torch.cat((torch.zeros((len(bg.xyz),3),device='cuda'),base_colors)))[0]
            photo=torch.tensor(cache['original_rgb'].transpose(2,0,1).copy(),device='cuda',dtype=torch.float32)/255
            reliable=torch.tensor(cache['background_train_mask'],device='cuda')
            bg_image=c.render(camera,bg,aux=False)['rgb']; denom=reliable.sum().clamp_min(1)*3
            outside=torch.tensor(~mask,device='cuda'); change=(colors[index]-colors[0]).abs().mean(0)
            vrow['stages'].append({'stage':('M3','M4_A','M4_B')[index],
                'visibility':float((visible*torch.tensor(mask,device='cuda')).sum()/(alpha*torch.tensor(mask,device='cuda')).sum().clamp_min(1)),
                'base_visibility':float(base_joint.sum()/base_alone.sum().clamp_min(1)),
                'base_visible_relative_to_m3':float(base_joint.sum()/reference_base.sum().clamp_min(1)),
                'visible_relative_to_m3_alpha':float((visible*torch.tensor(mask,device='cuda')).sum()/max(float((cache['object_alpha']*mask).sum()),1)),
                'real_background_l1':float(((bg_image-photo).abs()*reliable).sum()/denom),
                'legal_foreground_pixels':int(cache['legal_foreground'].sum()),
                'object_contribution_under_legal_foreground':float((visible*torch.tensor(cache['legal_foreground'],device='cuda')).sum()/torch.tensor(cache['legal_foreground'],device='cuda').sum().clamp_min(1)),
                'joint_outside_object_mae':float((change*outside).sum()/outside.sum().clamp_min(1)),
                'joint_outside_changed_005_fraction':float(((change>.05)&outside).sum()/outside.sum().clamp_min(1))})
        rows.append(vrow); print('validated',name,flush=True)
        if record['split']!='anchor':
            assert abs(vrow['stages'][1]['real_background_l1']-expected_a[name]['background_l1'])<2e-6, 'Exported SH colors differ from Stage A training render'
    panel=Image.new('RGB',(pages[0].width,sum(p.height for p in pages)),'white'); y=0
    for page in pages: panel.paste(page,(0,y)); y+=page.height
    panel.save(out/'multiview_overview.png')
    anchor=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    images=[c.render_joint(anchor,bg,obj,aux=False)['rgb'] for bg,obj in [(bg0,obj0),(bg1,obj0),(bg1,obj1)]]
    alpha_base=c.raster(anchor,obj0,base_colors)[0].cpu().numpy()
    y,x=np.where(alpha_base>.1); pad=round(min(anchor.width,anchor.height)*.03)
    zoom=(max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(anchor.width,int(x.max())+pad),min(anchor.height,int(y.max())+pad))
    board(images,['M3: original composition','M4 A: local background optimization','M4 B: DI-SDS appearance'],zoom).save(out/'same_view_comparison.png')
    write_json(out/'visibility_report.json',{'views':rows,'base_definition':'Lowest 20% of original object Gaussian centers along M1 world_up; same point membership in all stages',
        'stage_a_metrics':a,'unknown_regions':{'frozen_gaussians':manifest['local_unsupported_count'],
        'interpretation':'Insufficient support in the selected training photos, not proof of invisibility in every source video frame.',
        'per_view':[{'name':v['name'],'unknown_pixels':v['unknown_pixels'],'trusted_photo_pixels':v['local_reliable_pixels']} for v in manifest['views']]}})
    # Same calibrated cameras and unchanged model geometry in all video panels.
    with VideoWriter(out/'trajectory_m3_a_b.mp4',ctx.cfg['render']['fps']) as writer:
        for camera in c.source_cameras(640):
            frames=[pixels(c.render_joint(camera,bg,obj,aux=False)['rgb']) for bg,obj in [(bg0,obj0),(bg1,obj0),(bg1,obj1)]]
            writer.append_data(np.concatenate(frames,axis=1))
    preview_masks(ctx)
    preview_background(ctx)
    history=[json.loads(line) for line in (ctx.stage('m4')/'training.jsonl').read_text().splitlines()]
    write_json(out/'runtime_report.json',{'stage_a':{'iterations':a['iterations'],'selected_step':a['selected_step'],'seconds':a['seconds']},
        'stage_b':{'completed_steps':history[-1]['step'],'elapsed_seconds_this_attempt':history[-1]['elapsed_this_attempt'],
        'peak_gpu_gib':max(r['gpu_peak_gib'] for r in history),'mean_step_seconds':float(np.mean([r['seconds'] for r in history]))},
        'qwen_calls':0,'trellis_calls':0})
    write_json(out/'completed.json',{'all_checks_passed':True,'source_object_sha256':sha256(object_path),
        'stage_a_background_sha256':sha256(optimized_path),'final_scene_sha256':sha256(ctx.stage('m4')/'scene_final.ply')})


if __name__=='__main__':
    import argparse
    from ..core.context import Context
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--masks-only',action='store_true'); parser.add_argument('--background-only',action='store_true')
    args=parser.parse_args(); ctx=Context(args.config)
    (preview_masks if args.masks_only else preview_background if args.background_only else validate)(ctx)
