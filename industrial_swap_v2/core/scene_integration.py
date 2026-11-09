"""M4 owns alignment/composition for PBR-trained objects; old runs remain readable."""
from pathlib import Path
import numpy as np
from .context import read_json,write_json,sha256,gsle
from .pbr_pipeline import enabled


def object_path(ctx):
    return ctx.stage('m4' if enabled(ctx) else 'm3')/'object_world.ply'


def integrate(ctx):
    if not enabled(ctx): return object_path(ctx)
    from .scene import read_vertices,save_vertices,compose
    from .alignment import fit,fit_reference,transform_vertices
    from .m2_space import generation_frame
    src=ctx.stage('m3')/'object_refined_local.ply';out=ctx.stage('m4')
    summary=read_json(ctx.stage('m3')/'pbr_training_summary.json')
    if sha256(src)!=summary['object_refined_sha256']: raise ValueError('M3 optimized object changed')
    cfg=ctx.cfg['m4']['integration'];destination=out/'object_world.ply'
    if destination.exists():
        receipt=read_json(out/'integration.json')
        if receipt['source_sha256']!=sha256(src) or receipt['world_sha256']!=sha256(destination) or receipt['config']!=cfg:
            raise ValueError('Existing M4 integration does not match M3/config')
        return destination
    local=read_vertices(src);camera=read_json(ctx.stage('m2')/'anchor_camera.json')
    if cfg['alignment']=='verified_transform':
        path=ctx.source(cfg['existing_transform']);transform=read_json(path)
        if sha256(path.parent/'trellis_asset.glb')!=sha256(ctx.stage('m3')/'trellis_asset.glb'):
            raise ValueError('Historical transform belongs to another GLB')
        world=transform_vertices(local,float(transform['scale']),np.asarray(transform['rotation']),np.asarray(transform['translation']))
        save_vertices(destination,world)
        transform.update(method='verified placement reused for the identical GLB; all directional SH rotated',
                         source_transform=str(path),source_transform_sha256=sha256(path))
        write_json(out/'transform.json',transform)
    elif cfg['alignment']=='anchor_silhouette':
        fit(local,generation_frame(ctx),camera,ctx.stage('m2')/'object_mask.png',ctx.cfg['m3'],out)
    elif cfg['alignment']=='rasterized_reference':
        fit_reference(ctx)
    else: raise ValueError('Unknown M4 alignment method')
    compose(ctx.stage('m1')/'background_gaussians.ply',destination,out/'scene_initial.ply',out/'initial_composition.json')
    write_json(out/'integration.json',{'source':str(src),'source_sha256':sha256(src),'world_sha256':sha256(destination),
        'config':cfg,'gaussians':len(local),'sh_frame_rotation':True,'geometry_frozen_after_alignment':True})
    c=gsle(ctx)
    import torch
    obj=c.GaussianData.load(destination);bg=c.GaussianData.load(ctx.stage('m1')/'background_gaussians.ply')
    cam=c.Camera.from_record(camera);renders=out/'integration_renders';renders.mkdir(exist_ok=True)
    with torch.no_grad():
        c.save_image(renders/'anchor_object.png',c.render(cam,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'])
        c.save_image(renders/'anchor_scene.png',c.render_joint(cam,bg,obj,aux=False)['rgb'])
    del obj,bg;torch.cuda.empty_cache()
    return destination
