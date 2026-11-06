import subprocess
import os
from ..core.context import Context, arguments, read_json, sha256
from ..core.m3_composition import compose_m3
from ..core.rendering import render_outputs


def run(ctx):
    out=ctx.stage('m3')
    env=ctx.env()
    env.update(CUDA_HOME='/usr/local/cuda-12.4',HF_HOME=str(ctx.path('models/huggingface')),
        HF_MODULES_CACHE=str(ctx.path('models/huggingface/modules')),
        TORCH_HOME=str(ctx.path('models/torch')),TORCH_EXTENSIONS_DIR=str(ctx.path('tmp/torch_extensions')),
        TRITON_CACHE_DIR=str(ctx.path('tmp/triton')),XDG_CACHE_HOME=str(ctx.path('models/cache')),
        TMPDIR=str(ctx.path('tmp')),MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',TORCH_CUDA_ARCH_LIST='8.6')
    env['PATH']='/usr/local/cuda-12.4/bin:'+env['PATH']
    if all((out/n).is_file() for n in ('object_world.ply','object_local.ply','transform.json','trellis_asset.glb','trellis_asset_preview.png')):
        record=read_json(out/'trellis_generation.json')
        assert record['input_sha256']==sha256(ctx.path(ctx.cfg['m3']['input_image']))
        assert record['repository_commit']==ctx.cfg['m3']['revision']
        assert record['pipeline_type']==ctx.cfg['m3']['pipeline_type']
        print('Reusing saved TRELLIS and Gaussian assets; resuming scene export',flush=True)
    else:
        subprocess.run([str(ctx.source(ctx.cfg['project']['trellis_python'])),'-m',
            'industrial_swap_v2.core.trellis_worker','--config',str(ctx.config_path)],cwd=ctx.root,env=env,check=True)
    background=compose_m3(ctx)
    render_outputs(ctx,'m3',out/'object_world.ply',background_path=background)
    ctx.complete('m3',[out/n for n in ('trellis_asset.glb','trellis_asset_preview.png',
        'object_local.ply','object_world.ply','scene_coarse.ply','transform.json','renders/trajectory_before_after.mp4')])


if __name__=='__main__': run(Context(arguments().config))
