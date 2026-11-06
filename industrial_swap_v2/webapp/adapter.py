"""GPU-side adapters. No database dependency and no alternate generation algorithm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image
import numpy as np
from ..core.context import Context, read_json, write_json, sha256


def child(ctx, module, extra=(), trellis=False):
    env=ctx.env()
    if trellis:
        env.update(CUDA_HOME='/usr/local/cuda-12.4', HF_HOME=str(ctx.path('models/huggingface')),
            HF_MODULES_CACHE=str(ctx.path('models/huggingface/modules')), TORCH_HOME=str(ctx.path('models/torch')),
            TORCH_EXTENSIONS_DIR=str(ctx.path('tmp/torch_extensions')),TRITON_CACHE_DIR=str(ctx.path('tmp/triton')),
            TMPDIR=str(ctx.path('tmp')), MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.6')
        env['PATH']='/usr/local/cuda-12.4/bin:'+env['PATH']
    interpreter=ctx.cfg['project']['trellis_python' if trellis else 'sam_python']
    subprocess.run([str(ctx.source(interpreter)),'-m',module,'--config',str(ctx.config_path),*extra],
        cwd=ctx.root,env=env,check=True)


def image_import(ctx, job):
    from ..core.cameras import resized_camera
    out=ctx.stage('m2'); asset=job['asset']; source=Path(asset['path'])
    camera=read_json(out/'conditions.json')['camera']
    with Image.open(source) as original:
        image=original.convert('RGBA' if asset['kind']=='object_rgba' else 'RGB')
    if abs(image.width/image.height-camera['width']/camera['height'])>1e-6:
        raise ValueError('上传图像比例与参考画布不一致，不能隐式裁剪或填充')
    write_json(out/'anchor_camera.json',resized_camera(camera,image.width,image.height))
    if asset['kind']=='edited_scene':
        image.save(out/'edited_scene.png')
        child(ctx,'industrial_swap_v2.core.segmentation')
    else:
        image.save(out/'object_rgba.png')
        # Retain soft alpha, including antialiased boundaries, in reference supervision.
        image.getchannel('A').save(out/'object_mask.png')
        white=Image.new('RGBA',image.size,'white'); white.alpha_composite(image)
        white.convert('RGB').save(out/'object_white.png')
        write_json(out/'segmentation.json',{'method':'uploaded_alpha','sam_called':False})
    write_json(out/'generation.json',{'source':'asset_library','asset_id':asset['id'],
        'source_sha256':asset['sha256'],'external_api_called':False,'sam_called':asset['kind']=='edited_scene',
        'camera_alignment':'full anchor canvas, user-declared and aspect-validated'})
    ctx.complete('m2',[out/n for n in ('object_rgba.png','object_mask.png','object_white.png','anchor_camera.json','generation.json')])


def convert_asset(ctx, job):
    from ..core.alignment import fit, transform_vertices
    from ..core.scene import read_vertices, save_vertices
    from ..core.mesh_to_gaussian import convert, fields
    asset=job['asset']; meta=asset['metadata']; out=ctx.stage('m3')
    path=Path(asset['path']); kind=asset['kind']
    def canonical_vertices(path):
        original=read_vertices(path)
        result=np.empty(len(original),dtype=[(key,'f4') for key in fields()])
        for key in fields(): result[key]=original[key]
        return result
    if kind=='mesh':
        destination=out/('trellis_asset.glb' if path.suffix=='.glb' else 'imported_mesh'+path.suffix)
        shutil.copy2(path,destination)
        local=convert(path,out/'object_local.ply',ctx.cfg['m3'],ctx.cfg['project']['seed'])
    elif kind=='gaussian_local':
        local=canonical_vertices(path); save_vertices(out/'object_local.ply',local)
    elif kind=='gaussian_world':
        transformation=meta.get('transform')
        if not transformation: raise ValueError('世界坐标 PLY 缺少可验证的变换信息')
        same_scene=meta.get('scene_sha')==ctx.cfg['project']['source_sha256']
        if same_scene and not job['parameters'].get('realign',True):
            save_vertices(out/'object_world.ply',canonical_vertices(path)); write_json(out/'transform.json',transformation)
            return
        rot=np.asarray(transformation['rotation']); scale=float(transformation['scale']); shift=np.asarray(transformation['translation'])
        local=transform_vertices(canonical_vertices(path),1/scale,rot.T,-rot.T@shift/scale)
        save_vertices(out/'object_local.ply',local)
    else: raise ValueError('Unsupported model asset')
    fit(local,read_json(ctx.stage('m1')/'manifest.json')['world_transform'],
        read_json(ctx.stage('m2')/'anchor_camera.json'),ctx.stage('m2')/'object_mask.png',ctx.cfg['m3'],out)


def execute(ctx, job):
    stage,action=job['stage'],job['action']
    if action!='qwen': os.environ['SWAP_DISABLE_QWEN']='1'
    if action=='convert_asset':
        convert_asset(ctx,job); return
    if stage=='m1':
        if action=='run':
            from ..modules.m1_localization import run
            run(ctx)
        else:
            destination=ctx.stage('m1'); original=ctx.base/'outputs/m1'
            receipt=read_json(ctx.base/'records/m1_status.json')
            if not all(sha256(a['path'])==a['sha256'] for a in receipt['artifacts']):
                raise ValueError('历史 M1 产物完整性检查失败')
            for source in original.iterdir():
                if source.is_file():
                    target=destination/source.name
                    if target.exists(): continue
                    if source.suffix=='.json': shutil.copy2(source,target)
                    else: target.symlink_to(source)
            manifest=read_json(destination/'manifest.json')
            for key in ('target_gaussians','background_gaussians'):
                manifest[key]=str(destination/(key+'.ply'))
            write_json(destination/'manifest.json',manifest)
            ctx.complete('m1',[destination/n for n in ('manifest.json','target_gaussians.ply','background_gaussians.ply','target_indices.npy','keep_indices.npy')])
    elif stage=='m2':
        from ..modules.m2_image_generation import run
        if action=='qwen':
            authorization=read_json(Path(job['folder'])/'authorization.json')
            if not authorization.get('explicit_web_click'): raise ValueError('Missing explicit API authorization')
            os.environ.pop('SWAP_DISABLE_QWEN',None)
            run(ctx)
        else:
            run(ctx,prepare_only=True)
            if action=='reuse': image_import(ctx,job)
    elif stage=='m3':
        if action=='trellis' and not job.get('asset'):
            from ..modules.m3_trellis_generation import run
            run(ctx)
        else:
            child(ctx,'industrial_swap_v2.webapp.adapter',('--job',job['job_path'],'--convert-only'),trellis=True)
            from ..core.m3_composition import compose_m3
            from ..core.rendering import render_outputs
            out=ctx.stage('m3')
            background=compose_m3(ctx)
            render_outputs(ctx,'m3',out/'object_world.ply',background_path=background)
            write_json(out/'reuse.json',{'asset':job['asset']['id'],'trellis_called':False,'realign':job['parameters'].get('realign',True),
                'automatic_generation_cache':action=='trellis'})
            ctx.complete('m3',[out/n for n in ('object_world.ply','transform.json','scene_coarse.ply','renders/anchor_after.png')])
    elif stage=='m4':
        for name in ('object_rgba.png','object_white.png','object_mask.png','anchor_camera.json'):
            if not (ctx.stage('m2')/name).is_file(): raise ValueError('M4 缺少 '+name)
        from ..modules.m4_refinement import run
        run(ctx)
    else: raise ValueError('Unsupported stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--job',required=True)
    parser.add_argument('--convert-only',action='store_true'); args=parser.parse_args()
    context=Context(args.config); job=read_json(args.job)
    if args.convert_only: job['action']='convert_asset'
    execute(context,job)
    if not args.convert_only: write_json(Path(job['folder'])/'adapter_success.json',{'complete':True})
