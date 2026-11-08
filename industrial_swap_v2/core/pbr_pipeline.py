"""M3 orchestration: unchanged PBR asset -> Cycles views -> optimized local GS."""
import os
from pathlib import Path
import shutil
import subprocess
from .context import read_json,write_json,sha256

METHOD='pbr_multiview_gaussian'


def enabled(ctx): return getattr(ctx,'cfg',{}).get('m3',{}).get('method')==METHOD


def run(ctx):
    cfg=ctx.cfg['m3']['pbr'];out=ctx.stage('m3');glb=out/'trellis_asset.glb'
    source=cfg.get('source_glb')
    if source:
        source=ctx.source(source)
        if not source.is_file(): raise FileNotFoundError(source)
        if glb.exists():
            if sha256(glb)!=sha256(source): raise ValueError('Reused GLB mismatch')
        else: shutil.copy2(source,glb)
        write_json(out/'asset_provenance.json',{'source_glb':str(source),'sha256':sha256(glb),
            'trellis_invoked':False,'qwen_calls':0,'full_pbr_glb_preserved':True})
    elif not glb.exists():
        env=ctx.env();env.update(CUDA_HOME='/usr/local/cuda-12.4',HF_HUB_OFFLINE='1',
            HF_HOME=str(ctx.path('models/huggingface')),HF_MODULES_CACHE=str(ctx.path('models/huggingface/modules')),
            TORCH_HOME=str(ctx.path('models/torch')),TORCH_EXTENSIONS_DIR=str(ctx.path('tmp/torch_extensions')),
            TRITON_CACHE_DIR=str(ctx.path('tmp/triton')),XDG_CACHE_HOME=str(ctx.path('models/cache')),
            TMPDIR=str(ctx.path('tmp')),MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',TORCH_CUDA_ARCH_LIST='8.6')
        env['PATH']='/usr/local/cuda-12.4/bin:'+env['PATH']
        subprocess.run([str(ctx.source(ctx.cfg['project']['trellis_python'])),'-m',
            'industrial_swap_v2.core.trellis_worker','--config',str(ctx.config_path)],cwd=ctx.root,env=env,check=True)
    hdri=ctx.source(cfg['hdri'])
    if hdri.is_dir():
        candidates=[p for p in hdri.rglob('*') if p.suffix.lower() in ('.hdr','.exr')]
        if len(candidates)!=1: raise ValueError('HDRI directory must identify exactly one .hdr/.exr')
        hdri=candidates[0]
    if not hdri.is_file(): raise FileNotFoundError(hdri)
    dataset=out/'pbr_views';job={'glb':str(glb),'hdri':str(hdri),'output':str(dataset),
        'seed':ctx.cfg['project']['seed'],'render':cfg['render']}
    write_json(out/'render_job.json',job)
    if (dataset/'cameras.json').exists():
        saved=read_json(dataset/'cameras.json')
        if saved['glb_sha256']!=sha256(glb) or saved['hdri_sha256']!=sha256(hdri) or saved['config']!=cfg['render']:
            raise ValueError('PBR dataset fingerprint mismatch; use a new experiment')
    else:
        script=ctx.base/'scripts/blender_pbr_views.py'
        with (ctx.logs/'blender.log').open('a') as log:
            subprocess.run([str(ctx.source(cfg['blender'])),'--background','--python-exit-code','1',
                '--python',str(script),'--','--job',str(out/'render_job.json')],cwd=ctx.root,env=ctx.env(),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (dataset/'ray_depth_calibration.json').exists():
        from ..scripts.validate_pbr_calibration import validate
        validate(dataset)
    if not read_json(dataset/'ray_depth_calibration.json')['passed']:
        raise ValueError('PBR depth calibration failed')
    if not (out/'object_local.ply').exists():
        from .mesh_to_gaussian import convert
        convert(glb,out/'object_local.ply',ctx.cfg['m3'],ctx.cfg['project']['seed'])
    from .pbr_gaussian_optimizer import optimize
    report=optimize(ctx)
    ctx.complete('m3',[out/name for name in ('trellis_asset.glb','object_local.ply','object_refined_local.ply',
        'pbr_views/cameras.json','pbr_views/dataset_validation.json','pbr_training_summary.json',
        'checkpoints/best.pth','renders/validation_comparison.jpg')],
        method=METHOD,training_steps=report['iterations'],best_step=report['best_step'],alignment_deferred_to='m4')
