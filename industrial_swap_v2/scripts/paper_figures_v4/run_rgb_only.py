"""Matched-budget Case00 RGB-only ablation, fixed object and existing masks."""
import sys,os
from pathlib import Path
import yaml,torch
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,read_json,write_json
import industrial_swap_v2.core.local_background_optimizer as opt

def main():
    row=read_json(ROOT/'baseline_comparison/prompts.json')['cases'][0]
    original=Path(row['ours_source']);base=ROOT/'industrial_swap_v2'
    dest=base/'outputs/paper_figures_v4/F14_background_refinement/rgb_only_run'
    assert not dest.exists();dest.mkdir(parents=True)
    cfg=yaml.safe_load((original/'config.yaml').read_text())
    cfg['project'].update(output_dir=str(dest.relative_to(base)),records_dir=str((dest/'records').relative_to(base)),logs_dir=str((dest/'logs').relative_to(base)),gpu=0)
    cfg['m4'].pop('resume',None);cfg['m4']['save_every']=2000
    cfg['m4']['local_background'].update(iterations=2000,visibility_weight=0.,occlusion_environment_guard_weight=0.,legal_foreground_guard_weight=0.,occlusion_regularization=0.)
    for stage in ['m1','m2','m3']:(dest/stage).symlink_to((original/stage).resolve(),target_is_directory=True)
    (dest/'m4').mkdir();(dest/'m4/object_world.ply').symlink_to(original/'m4/object_world.ply')
    (dest/'m4/object_fixed.ply').symlink_to(original/'m4/object_world.ply')
    (dest/'m4/multiview_supervision').symlink_to(original/'m4/multiview_supervision',target_is_directory=True)
    p=dest/'config.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False));ctx=Context(p)
    # Reuse the exact original supervision/cache, not a different mask selection.
    opt.ensure_m4_supervision=lambda context:original/'m4/multiview_supervision'
    torch.cuda.set_per_process_memory_fraction(.35)
    result=opt.optimize_background(ctx)
    write_json(dest/'ablation.json',dict(source_run=str(original),iterations=2000,
        fixed_object=True,same_masks=True,same_learning_rates=True,same_view_seed=True,
        disabled_losses=['visibility','occlusion_environment_guard','legal_foreground_guard','occlusion_regularization'],
        retained_losses=['environment_RGB','color_regularization','opacity_regularization'],
        guarded_background=str(result),status='completed'))

if __name__=='__main__':main()
