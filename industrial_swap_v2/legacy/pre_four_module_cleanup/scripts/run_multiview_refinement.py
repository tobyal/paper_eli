"""Isolated existing-asset experiment; never imports Qwen or TRELLIS workers."""
import argparse
import os
from pathlib import Path
import shutil
import yaml
from ..core.context import Context, sha256, write_json


def initialize(source_config, destination, gpu):
    source=Context(source_config); base=source.base
    root=(base/destination).resolve(); root.relative_to(base.resolve())
    config=root/'config.yaml'
    if config.exists(): return config
    root.mkdir(parents=True,exist_ok=True)
    cfg=source.cfg
    cfg['project'].update(output_dir=str(root.relative_to(base)),gpu=gpu,
        records_dir=str((root/'records').relative_to(base)),logs_dir=str((root/'logs').relative_to(base)))
    cfg['m3']['object_occlusion_filter']={'enabled':False}
    cfg['m3']['visibility_initialization']={'enabled':True,'resolution':640,'source_views':40}
    cfg['m4']['local_background']={'enabled':True,'iterations':1000}
    if cfg['m4'].get('background_only',False):
        cfg['m3']['visibility_initialization']['scheme']='old_new_union_v2'
        cfg['m4']['local_background']['iterations']=250
    cfg['m4']['diffusion_mask_source']='object_alpha'
    cfg['m4']['object_mask_dilation_fraction']=.015
    cfg['m4']['initialize_from']=str((root/'m3/object_world.ply').relative_to(base))
    for stage in ('m1','m2'):
        (root/stage).symlink_to(source.stage(stage).resolve(),target_is_directory=True)
    (root/'m3').mkdir()
    sources={}
    for name in ('object_world.ply','transform.json'):
        path=source.stage('m3')/name
        sources[str(path)]=sha256(path); shutil.copy2(path,root/'m3'/name)
    for path in (source.stage('m1')/'background_gaussians.ply',source.stage('m2')/'object_mask.png',source.stage('m2')/'object_rgba.png'):
        sources[str(path)]=sha256(path)
    write_json(root/'source_preservation.json',sources)
    for name,target in {'m3_visibility':'m3/multiview_visibility','m4_background':'m4/background_optimization','m4_refined':'m4'}.items():
        (root/name).symlink_to(target,target_is_directory=True)
    config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    return config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-config',type=Path)
    parser.add_argument('--destination',default='outputs/multiview_scene_refinement/run_v1')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--stage',choices=['prepare','background','refine','validate','all'],required=True)
    args=parser.parse_args()
    config=args.config or initialize(args.source_config,args.destination,args.gpu)
    ctx=Context(config); os.environ.update(ctx.env())
    os.environ.update(HF_HOME=str(ctx.path('models/huggingface')),TORCH_HOME=str(ctx.path('models/torch')),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if args.stage in ('prepare','all'):
        from ..core.scene import compose
        from ..core.multiview_masks import prepare_multiview
        bg=ctx.stage('m1')/'background_gaussians.ply'; obj=ctx.stage('m3')/'object_world.ply'
        compose(bg,obj,ctx.stage('m3')/'scene_coarse.ply',ctx.stage('m3')/'composition.json')
        write_json(ctx.stage('m3')/'background_selection.json',{'filter_enabled':False,'mode':'multiview_initialization','sha256':sha256(bg)})
        visibility=prepare_multiview(ctx)
        ctx.complete('m3',[obj,ctx.stage('m3')/'scene_coarse.ply',visibility/'manifest.json',visibility/'occlusion_candidates.json'],
            reused_aligned_asset=True,generator_invoked=False,background_deleted_points=0)
    if args.stage in ('background','all'):
        from ..core.local_background_optimizer import optimize_background
        optimize_background(ctx)
    if args.stage in ('refine','all'):
        from ..modules.m4_refinement import run
        run(ctx)
    if args.stage in ('validate','all'):
        if ctx.cfg['m4'].get('background_only',False):
            from .run_background_only import validate_background_only
            validate_background_only(ctx)
        else:
            from .validate_multiview_refinement import validate
            validate(ctx)


if __name__=='__main__': main()
