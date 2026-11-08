"""Bounded, existing-asset background optimization; no generative stage exists here."""
import argparse
import os
from pathlib import Path
import shutil
import yaml
from ..core.context import Context, read_json, write_json, sha256
from ..core.m4_outputs import export_m4
from ..core.workflow import current_config,validate_workflow


def initialize(source_config,destination,gpu,steps,save_every=500,resume_checkpoint=None):
    if not 200<=steps<=10000: raise ValueError('M4 must be 200–10000 steps')
    source=Context(source_config); root=source.path(destination)
    if root.exists(): raise FileExistsError('Use a new output directory; historical outputs are immutable')
    root.mkdir(parents=True); cfg=current_config(source.cfg)
    cfg['project'].update(output_dir=str(root.relative_to(source.base)),gpu=gpu,
        records_dir=str((root/'records').relative_to(source.base)),logs_dir=str((root/'logs').relative_to(source.base)))
    cfg['m3']['input_image']=str((root/'m2/object_rgba.png').relative_to(source.base))
    cfg['m4']['local_background']['iterations']=steps
    cfg['m4']['save_every']=save_every
    if resume_checkpoint:
        checkpoint=source.path(str(resume_checkpoint))
        if not checkpoint.is_relative_to(source.out): raise ValueError('Resume checkpoint must belong to the source experiment')
        cfg['m4']['resume']={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
            'source_config':str(source.config_path),'source_config_sha256':sha256(source.config_path)}
    validate_workflow(cfg)
    for stage in ('m1','m2'): (root/stage).symlink_to(source.stage(stage).resolve(),target_is_directory=True)
    (root/'m3').mkdir()
    for name in ('object_world.ply','transform.json'): shutil.copy2(source.stage('m3')/name,root/'m3'/name)
    caches=[source.stage('m4')/'multiview_supervision',source.stage('m4')/'background_optimization/visibility_prepare',
            source.stage('m3')/'multiview_visibility']
    cached=next((p for p in caches if (p/'manifest.json').is_file()),None)
    if cached:
        (root/'m4').mkdir()
        # M4 validates all source hashes/config before reusing this read-only cache.
        (root/'m4/multiview_supervision').symlink_to(cached.resolve(),target_is_directory=True)
    originals=[source.stage('m3')/'object_world.ply',source.stage('m1')/'background_gaussians.ply',
        source.stage('m2')/'object_mask.png',source.stage('m2')/'object_white.png',source.stage('m2')/'anchor_camera.json']
    write_json(root/'source_preservation.json',{str(p.resolve()):sha256(p) for p in originals})
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    return root/'config.yaml'




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-config',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--destination',help='New output directory relative to industrial_swap_v2')
    parser.add_argument('--steps',type=int,default=250)
    parser.add_argument('--save-every',type=int,default=500)
    parser.add_argument('--resume-checkpoint',type=Path,help='Actual snapshot checkpoint; --steps is the total target iteration')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--stage',choices=('prepare','background','validate','all'),default='all')
    args=parser.parse_args()
    if not args.config and (not args.source_config or not args.destination):
        parser.error('Use --config, or both --source-config and --destination')
    if args.config and args.resume_checkpoint: parser.error('Use --resume-checkpoint with --source-config and a new --destination')
    config=args.config or initialize(args.source_config,args.destination,args.gpu,args.steps,args.save_every,args.resume_checkpoint)
    ctx=Context(config); os.environ.update(ctx.env()); os.environ['SWAP_DISABLE_QWEN']='1'
    validate_workflow(ctx.cfg)
    if args.stage in ('prepare','all'):
        from ..core.multiview_masks import ensure_m4_supervision
        ensure_m4_supervision(ctx)
    if args.stage in ('background','all'):
        from ..core.local_background_optimizer import optimize_background
        optimize_background(ctx)
    if args.stage in ('validate','all'): export_m4(ctx)


if __name__=='__main__': main()
