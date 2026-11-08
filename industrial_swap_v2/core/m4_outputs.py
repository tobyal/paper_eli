"""Export guarded M4 scene and separately labelled last-step candidate."""
from pathlib import Path
import shutil
from .context import read_json,write_json,sha256
from .scene_integration import object_path


def compose_final_scene(ctx,summary):
    """The default scene always uses the integrity-checked, guarded background."""
    from .m3_composition import effective_background
    from .scene import compose
    stage=ctx.stage('m4'); obj=object_path(ctx)
    if sha256(obj)!=summary['object_sha256']: raise ValueError('Fixed device changed')
    background=effective_background(ctx,'m4')
    compose(background,obj,stage/'scene_final.ply',stage/'composition.json')
    shutil.copy2(obj,stage/'object_fixed.ply')


def export_m4(ctx):
    import numpy as np
    import torch
    from PIL import Image
    from .context import gsle
    from .scene import read_vertices,save_vertices,compose
    from .visibility_loss import object_contribution
    from .preview_panels import board,panel
    c=gsle(ctx); stage=ctx.stage('m4'); out=stage/'validation'; out.mkdir(exist_ok=True)
    training=stage/'background_optimization'; summary=read_json(training/'summary.json')
    folder=Path(summary['visibility_folder']); manifest=read_json(folder/'manifest.json')
    original=ctx.stage('m1')/'background_gaussians.ply'; objpath=object_path(ctx)
    # Always expose the actual last trained result, even if a conservative
    # checkpoint guard retained the input. Never conceal an unsuccessful guard.
    saved=torch.load(training/'checkpoint_latest.pth',map_location='cpu',weights_only=False)
    assert saved['signature']==summary['signature'] and saved['step']==summary['iterations']
    state=saved['state']; raw=read_vertices(original).copy()
    ids=state['indices'].numpy(); rgb_ids=state['rgb_indices'].numpy()
    raw['opacity'][ids]=(state['initial_logits']+state['opacity_delta']).numpy()[:,0]
    for channel in range(3): raw[f'f_dc_{channel}'][rgb_ids]+=state['dc_delta'].numpy()[:,channel]
    optimized=training/f'background_candidate_step{saved["step"]}.ply'
    save_vertices(optimized,raw); del raw,state,saved
    candidate_scene=stage/f'scene_candidate_step{summary["iterations"]}.ply'
    compose(optimized,objpath,candidate_scene,stage/'candidate_composition.json')
    evaluations=read_json(training/'optimization_log.json')['evaluations']
    candidate_metrics=evaluations[-1]
    compose_final_scene(ctx,summary)
    original_vertices=read_vertices(original); changed_vertices=read_vertices(optimized); object_vertices=read_vertices(objpath)
    scene=read_vertices(candidate_scene); indices=np.load(training/'optimized_indices.npy')
    frozen=np.ones(len(original_vertices),bool); frozen[indices]=False
    assert original_vertices[frozen].tobytes()==changed_vertices[frozen].tobytes()
    assert scene[:len(changed_vertices)].tobytes()==changed_vertices.tobytes()
    assert scene[len(changed_vertices):].tobytes()==object_vertices.tobytes()
    assert sha256(stage/'object_fixed.ply')==sha256(objpath)==summary['object_sha256']
    for name in original_vertices.dtype.names:
        if name not in ('opacity','f_dc_0','f_dc_1','f_dc_2'): assert original_vertices[name].tobytes()==changed_vertices[name].tobytes()
    non_rgb=np.ones(len(original_vertices),bool); non_rgb[rgb_ids]=False
    for k in range(3): assert original_vertices[f'f_dc_{k}'][non_rgb].tobytes()==changed_vertices[f'f_dc_{k}'][non_rgb].tobytes()
    # Available for all CLI/Web runs, including reused immutable M3 inputs.
    for name,digest in summary['source_hashes'].items():
        assert sha256(manifest['sources'][name])==digest,name
    preservation=ctx.out/'source_preservation.json'
    if preservation.exists():
        for path,digest in read_json(preservation).items(): assert sha256(path)==digest,path
    del original_vertices,changed_vertices,object_vertices,scene
    bg0=c.GaussianData.load(original); bg1=c.GaussianData.load(optimized); obj=c.GaussianData.load(objpath)
    up=torch.tensor(read_json(ctx.stage('m1')/'manifest.json')['world_up'],device='cuda')
    height=obj.xyz@up; base=(height<torch.quantile(height,.2))[:,None].float().expand(-1,3).contiguous()
    exported=[v for v in manifest['views'] if v['split'] in ('anchor','validation')]
    # Include strongest observed legal-foreground cases even when they are train views.
    for v in sorted(manifest['views'],key=lambda x:x['legal_foreground_pixels'],reverse=True)[:3]:
        if v['name'] not in {r['name'] for r in exported}: exported.append(v)
    for metric in sorted(candidate_metrics['views'],key=lambda x:x['outside_union_mae'],reverse=True)[:2]:
        if metric['name'] not in {r['name'] for r in exported}:
            exported.append(next(v for v in manifest['views'] if v['name']==metric['name']))
    rows=[]
    with torch.no_grad():
        for record in exported:
            camera=c.Camera.from_record(record['camera']); name=record['name']; cache=np.load(folder/'cache'/f'{name}.npz')
            before=c.render_joint(camera,bg0,obj,aux=False)['rgb']; after=c.render_joint(camera,bg1,obj,aux=False)['rgb']
            alone_base=c.raster(camera,obj,base)[0]
            colors=torch.cat((torch.zeros((len(bg0.xyz),3),device='cuda'),base))
            base_before=c.raster(camera,c.concatenate(bg0,obj),colors)[0]; base_after=c.raster(camera,c.concatenate(bg1,obj),colors)[0]
            visible_before=object_contribution(c,camera,bg0,obj)[0]; visible_after=object_contribution(c,camera,bg1,obj)[0]
            env=torch.tensor(cache['environment_mask'],device='cuda'); legal=torch.tensor(cache['legal_foreground'],device='cuda')
            mask=torch.tensor(cache['new_mask'],device='cuda'); alpha=torch.tensor(cache['object_alpha'],device='cuda')
            diff=(before-after).abs().mean(0)
            denom=(alpha*mask).sum().clamp_min(1)
            row={'name':name,'split':record['split'],
                'object_visible_before':float((visible_before*mask).sum()/denom),'object_visible_after':float((visible_after*mask).sum()/denom),
                'base_visible_before':float(base_before.sum()/alone_base.sum().clamp_min(1)),
                'base_visible_after':float(base_after.sum()/alone_base.sum().clamp_min(1)),
                'outside_union_joint_mae':float((diff*env).sum()/env.sum().clamp_min(1)),
                'outside_union_changed_005':float(((diff>.05)*env).sum()/env.sum().clamp_min(1)),
                'legal_foreground_pixels':int(legal.sum()),
                'legal_foreground_joint_mae':float((diff*legal).sum()/legal.sum().clamp_min(1)),
                'object_leak_in_legal_foreground_before':float((visible_before*legal).sum()/legal.sum().clamp_min(1)),
                'object_leak_in_legal_foreground_after':float((visible_after*legal).sum()/legal.sum().clamp_min(1))}
            rows.append(row)
            ys,xs=np.where(alone_base.cpu().numpy()>.1)
            zoom=None if len(xs)==0 else (max(0,int(xs.min())-15),max(0,int(ys.min())-15),min(camera.width,int(xs.max())+15),min(camera.height,int(ys.max())+15))
            image=board([before,after],['Before: original M1 + fixed M3',f'Candidate: fixed object, step {summary["iterations"]}'],zoom)
            image.save(out/f'{name}_comparison.png')
            c.save_image(out/f'{name}_before.png',before); c.save_image(out/f'{name}_after.png',after)
            c.save_image(out/f'{name}_outside_union_change.png',diff*env)
            c.save_image(out/f'{name}_visible_before.png',visible_before); c.save_image(out/f'{name}_visible_after.png',visible_after)
            if record['split']=='anchor': image.save(out/'anchor_comparison.png')
    # Actual train-camera masks, with no bounding-box exclusion.
    ac=manifest['views'][0]['camera']; ar=np.asarray(ac['R']); ap=-ar.T@np.asarray(ac['t'])
    def pose_distance(v):
        r=np.asarray(v['camera']['R']); p=-r.T@np.asarray(v['camera']['t'])
        return np.linalg.norm(p-ap)/manifest['region_scale']+np.linalg.norm(r-ar)
    example=min((v for v in manifest['views'] if v['split']=='train'),key=pose_distance)
    name=example['name']; cache=np.load(folder/'cache'/f'{name}.npz')
    images=[Image.fromarray(cache['original_rgb'])]+[Image.fromarray(cache[k].astype('uint8')*255) for k in ('old_mask','new_mask','environment_mask')]
    panel(out/'supervision_union_masks.png',images,
        ['原始真实图像','旧设备可见 Mask：M_old','新设备独立渲染 Mask：M_new','环境监督：1 − (M_old ∪ M_new)'],
        [example['camera']['name'],'原始场景中 M1 目标高斯的可见贡献','固定设备全部参数；每个相机独立渲染','白色为环境区域；训练另加小幅边界保护'],640,
        '新的环境监督：仅排除旧、新设备轮廓，不排除大包围盒','四图使用同一原始相机；没有补全图像或重新生成模型。')
    checks={'object_file_sha256_unchanged':True,'object_scene_rows_bitwise_identical':True,
        'unselected_background_bitwise_identical':True,'background_geometry_bitwise_identical':True,
        'non_rgb_background_colors_bitwise_identical':True,
        'original_sources_sha256_unchanged':True,'background_added':0,'background_deleted':0,
        'background_count':len(bg0.xyz),'object_count':len(obj.xyz),
        'di_sds_executed':False,'qwen_calls':0,'trellis_calls':0,'training_steps':summary['iterations']}
    report={'status':'completed_candidate_requires_review' if not candidate_metrics['acceptable'] else 'completed_guard_passed',
        'run':str(ctx.out),'checks':checks,'rendered_step':summary['iterations'],
        'selected_step':summary['selected_step'],'candidate_guard_passed':candidate_metrics['acceptable'],
        'candidate_background':str(optimized),'candidate_scene':str(candidate_scene),
        'guarded_background':str(training/'background_optimized.ply'),'guarded_scene':str(stage/'scene_final.ply'),
        'candidate_sha256':{'background':sha256(optimized),'scene':sha256(candidate_scene),'object':sha256(objpath)},
        'rgb_points':summary['rgb_points'],
        'occlusion_points':summary['occlusion_points'],'occlusion_without_rgb_points':summary['occlusion_without_rgb_points'],
        'baseline':summary['baseline']['summary'],'candidate':candidate_metrics['summary'],
        'guarded_final':summary['final']['summary'],'views':rows,
        'base_definition':'Same lowest 20% of fixed object Gaussian centers along M1 world_up in every render.',
        'limitations':'Foreground protection uses reconstruction depth, covariance support and multiview attribution, not human-labelled foreground truth. No unseen-ground completion.'}
    write_json(out/'report.json',report)
    write_json(stage/'summary.json',report)
    renders=stage/'renders'; renders.mkdir(exist_ok=True)
    anchor_name=manifest['views'][0]['name']
    shutil.copy2(out/f'{anchor_name}_before.png',renders/'anchor_before.png')
    shutil.copy2(out/f'{anchor_name}_after.png',renders/f'anchor_candidate_step{summary["iterations"]}.png')
    # The default preview must match the guarded scene_final, not the candidate.
    if summary['selected_step']==0:
        shutil.copy2(out/f'{anchor_name}_before.png',renders/'anchor_after.png')
    elif summary['selected_step']==summary['iterations']:
        shutil.copy2(out/f'{anchor_name}_after.png',renders/'anchor_after.png')
    else:
        guarded=c.GaussianData.load(training/'background_optimized.ply')
        with torch.no_grad():
            camera=c.Camera.from_record(manifest['views'][0]['camera'])
            c.save_image(renders/'anchor_after.png',c.render_joint(camera,guarded,obj,aux=False)['rgb'])
    ctx.complete('m4',[optimized,training/'background_optimized.ply',stage/'object_fixed.ply',
        candidate_scene,stage/'scene_final.ply',stage/'summary.json',out/'report.json',
        out/'anchor_comparison.png',renders/'anchor_after.png'],
        background_only=True,di_sds_executed=False,iterations=summary['iterations'],candidate_guard_passed=candidate_metrics['acceptable'])
    print('M4 local background optimization and export finished',flush=True)
    print(rows[0],flush=True)
