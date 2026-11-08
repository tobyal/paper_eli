"""Immutable iteration exports of the actual background state, before best selection."""
from pathlib import Path
import numpy as np
import torch
from .context import sha256,write_json
from .scene import read_vertices,save_vertices,compose
from .scene_integration import object_path as integrated_object


def export_background(model,source,destination):
    original=read_vertices(source); updated=original.copy()
    ids=model.indices.detach().cpu().numpy(); rgb=model.rgb_indices.detach().cpu().numpy()
    updated['opacity'][ids]=(model.initial_logits+model.opacity_delta).detach().cpu().numpy()[:,0]
    delta=model.dc_delta.detach().cpu().numpy()
    for k in range(3): updated[f'f_dc_{k}'][rgb]+=delta[:,k]
    save_vertices(destination,updated)
    check=read_vertices(destination); frozen=np.ones(len(original),bool); frozen[ids]=False
    assert original[frozen].tobytes()==check[frozen].tobytes()
    for name in original.dtype.names:
        if name not in ('opacity','f_dc_0','f_dc_1','f_dc_2'):
            assert original[name].tobytes()==check[name].tobytes()
    no_rgb=np.ones(len(original),bool); no_rgb[rgb]=False
    for k in range(3): assert original[f'f_dc_{k}'][no_rgb].tobytes()==check[f'f_dc_{k}'][no_rgb].tobytes()
    return {'background_points':len(original),'optimized_points':len(ids),
        'device_parameters_fixed':True,'background_geometry_bitwise_preserved':True,
        'unselected_background_bitwise_preserved':True,'non_rgb_colors_bitwise_preserved':True,
        'added_points':0,'deleted_points':0}


@torch.no_grad()
def save_iteration_result(ctx,model,obj,views,common,step,metrics,optimizer,rng,order,signature,best,best_step):
    from .preview_panels import board
    out=ctx.stage('m4')/'snapshots'/f'iteration_{step:06d}'
    temporary=out.with_name(out.name+'.incomplete')
    if out.exists() or temporary.exists(): raise FileExistsError('Snapshot already exists: '+str(out))
    temporary.mkdir(parents=True)
    source=ctx.stage('m1')/'background_gaussians.ply'; object_path=integrated_object(ctx)
    object_hash=sha256(object_path)
    checks=export_background(model,source,temporary/'background_optimized.ply')
    compose(temporary/'background_optimized.ply',object_path,temporary/'scene.ply',temporary/'composition.json')
    scene=read_vertices(temporary/'scene.ply'); device=read_vertices(object_path)
    assert scene[checks['background_points']:].tobytes()==device.tobytes()
    checks['object_scene_rows_bitwise_identical']=True
    del scene,device
    # Include the optimizer and all sampling states for a reproducible restart.
    torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
        'signature':signature,'rng':rng.bit_generator.state,'remaining_view_order':list(order),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'best_state':best,'best_step':best_step},temporary/'checkpoint.pth')
    write_json(temporary/'metrics.json',metrics)
    # Use the exported SH0 colors, not the frozen source SH in model.data().
    bg=common.GaussianData.load(temporary/'background_optimized.ply')
    validation=[v for v in views if v['split']=='validation']
    chosen=[views[0],*[validation[i] for i in np.unique(np.linspace(0,len(validation)-1,min(3,len(validation))).astype(int))]]
    renders=temporary/'renders'; renders.mkdir()
    for view in chosen:
        camera=view['camera_object']; name=view['name']
        before=common.render_joint(camera,model.original,obj,aux=False)['rgb']
        after=common.render_joint(camera,bg,obj,aux=False)['rgb']
        common.save_image(renders/f'{name}_before.png',before)
        common.save_image(renders/f'{name}_after.png',after)
        comparison=board([before,after],['Original M1 + fixed M3',f'M4 actual iteration {step}'])
        comparison.save(renders/f'{name}_comparison.png')
        if view['split']=='anchor': comparison.save(temporary/'anchor_comparison.png')
    assert sha256(object_path)==object_hash
    files={str(p.relative_to(temporary)):sha256(p) for p in temporary.rglob('*') if p.is_file()}
    write_json(temporary/'snapshot.json',{'step':step,'kind':'actual_iteration_not_guarded_selection',
        'guard_passed':bool(metrics['acceptable']),'object_path':str(object_path),'object_sha256':object_hash,
        'source_background_sha256':sha256(source),'checks':checks,
        'cameras':[v['camera'] for v in chosen],'files_sha256':files})
    temporary.rename(out)
    print('Saved M4 iteration snapshot',step,str(out),flush=True)
    del bg; torch.cuda.empty_cache()
    return out
