"""One background selection for M3 export, Web reuse, and downstream M4."""
from dataclasses import asdict
from pathlib import Path
from .context import gsle, read_json, write_json, sha256


def effective_background(ctx,stage='m3'):
    if stage=='m4' and ctx.cfg['m4'].get('local_background',{}).get('enabled',False):
        folder=ctx.stage('m4')/'background_optimization'
        summary=read_json(folder/'summary.json'); path=folder/'background_optimized.ply'
        if sha256(path)!=summary['background_sha256']: raise ValueError('Stage A background integrity mismatch')
        return path
    selection=ctx.stage('m3')/'background_selection.json'
    if not selection.exists(): return ctx.stage('m1')/'background_gaussians.ply'
    record=read_json(selection)
    if not record['filter_enabled']: return ctx.stage('m1')/'background_gaussians.ply'
    path=(ctx.stage('m3')/record['relative_path']).resolve()
    path.relative_to(ctx.stage('m3').resolve())
    if sha256(path)!=record['sha256']: raise ValueError('Filtered background integrity mismatch')
    return path


def compose_m3(ctx):
    from .scene import compose
    out=ctx.stage('m3'); background=ctx.stage('m1')/'background_gaussians.ply'; obj=out/'object_world.ply'
    if ctx.cfg['m3'].get('visibility_initialization',{}).get('enabled',False):
        if ctx.cfg['m3'].get('object_occlusion_filter',{}).get('enabled',False):
            raise ValueError('Multiview initialization and archived deletion filtering cannot be enabled together')
        from .multiview_masks import prepare_multiview
        write_json(out/'background_selection.json',{'filter_enabled':False,'mode':'multiview_initialization','sha256':sha256(background)})
        compose(background,obj,out/'scene_coarse.ply',out/'composition.json')
        prepare_multiview(ctx)
        return background
    from .object_occlusion_filter import filter_config, filter_object_occlusion, spatial_context_digest
    cfg=filter_config(ctx.cfg['m3'].get('object_occlusion_filter'))
    if cfg.enabled:
        c=gsle(ctx); mask=ctx.stage('m2')/'object_mask.png'
        anchor=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
        cameras=[anchor,*c.source_cameras(cfg.resolution)]
        region=read_json(ctx.stage('m1')/'manifest.json')
        folder=out/'occlusion_filter'; report_path=folder/'occlusion_report.json'
        if report_path.exists():
            report=read_json(report_path)
            matching=report['config']==asdict(cfg) and report['input_sha256']=={
                'background':sha256(background),'object':sha256(obj),'reference_mask':sha256(mask)}
            matching&=all(sha256(folder/name)==digest for name,digest in report['output_sha256'].items())
            matching&=report.get('spatial_context_sha256')==spatial_context_digest(region,cameras)
        else: matching=False
        if not matching:
            attempt=1
            while folder.exists():
                attempt+=1; folder=out/f'occlusion_filter_{attempt:03d}'
            filter_object_occlusion(background,obj,region,mask,cameras,folder,asdict(cfg))
        background=folder/'background_filtered.ply'
        selection={'filter_enabled':True,'relative_path':str(background.relative_to(out)),
                   'sha256':sha256(background),'report':str((folder/'occlusion_report.json').relative_to(out))}
    else: selection={'filter_enabled':False,'source':'original_m1_background','sha256':sha256(background)}
    write_json(out/'background_selection.json',selection)
    compose(background,obj,out/'scene_coarse.ply',out/'composition.json')
    return background
