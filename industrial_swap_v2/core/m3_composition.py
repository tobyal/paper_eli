"""M3 uses original M1 background; M4 uses the guarded optimized background."""
from .context import read_json,write_json,sha256
from .workflow import validate_workflow


def effective_background(ctx,stage='m3'):
    if stage=='m4':
        folder=ctx.stage('m4')/'background_optimization'
        summary=read_json(folder/'summary.json'); path=folder/'background_optimized.ply'
        if sha256(path)!=summary['background_sha256']:
            raise ValueError('M4 background integrity mismatch')
        return path
    return ctx.stage('m1')/'background_gaussians.ply'


def compose_m3(ctx):
    validate_workflow(ctx.cfg,stage='m3')
    from .scene import compose
    out=ctx.stage('m3'); background=effective_background(ctx)
    write_json(out/'background_selection.json',{'filter_enabled':False,
        'mode':'scene_composition','source':'original_m1_background','sha256':sha256(background)})
    compose(background,out/'object_world.ply',out/'scene_coarse.ply',out/'composition.json')
    return background
