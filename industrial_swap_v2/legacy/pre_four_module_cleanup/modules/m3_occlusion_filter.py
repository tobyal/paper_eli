"""Filter an existing M3 into a new directory, with no model generation."""
import argparse
from dataclasses import asdict
import yaml
from ..core.context import Context, gsle, read_json, write_json, sha256
from ..core.object_occlusion_filter import filter_config, filter_object_occlusion


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--output',required=True); p.add_argument('--settings')
    args=p.parse_args(); ctx=Context(args.config); c=gsle(ctx)
    settings=dict(ctx.cfg['m3'].get('object_occlusion_filter',{})); settings['enabled']=True
    if args.settings:
        from pathlib import Path
        settings.update(yaml.safe_load(Path(args.settings).read_text(encoding='utf-8')))
    cfg=filter_config(settings)
    watched=[ctx.stage('m1')/name for name in ('manifest.json','edit_region.json','background_gaussians.ply')]
    watched += [ctx.stage('m2')/name for name in ('object_mask.png','anchor_camera.json')]
    watched += [ctx.stage('m3')/name for name in ('object_world.ply','transform.json','scene_coarse.ply')]
    # Existing historical M1–M4 receipts include the major successful outputs.
    for stage in ('m1','m2','m3','m4'):
        receipt=ctx.base/'records'/f'{stage}_status.json'
        if receipt.exists():
            watched.append(receipt)
            watched += [ctx.source(a['path']) for a in read_json(receipt).get('artifacts',[]) if ctx.source(a['path']).is_file()]
    watched=sorted({p.resolve() for p in watched}); before={str(p):sha256(p) for p in watched}
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    out=ctx.path(args.output)
    filter_object_occlusion(ctx.stage('m1')/'background_gaussians.ply',ctx.stage('m3')/'object_world.ply',
        read_json(ctx.stage('m1')/'manifest.json'),ctx.stage('m2')/'object_mask.png',
        [camera,*c.source_cameras(cfg.resolution)],out,asdict(cfg))
    after={str(p):sha256(p) for p in watched}
    assert after==before,'An existing experiment output changed'
    write_json(out/'preservation.json',{'unchanged':True,'sha256':after,'qwen_calls':0,'trellis_calls':0})


if __name__=='__main__': main()
