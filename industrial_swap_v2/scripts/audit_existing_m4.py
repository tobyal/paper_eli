"""Read-only compatibility audit of a completed union run; never renders/trains."""
import argparse
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
from ..core.context import Context,read_json,sha256
from ..core.local_background_optimizer import optimize_background
from ..core.scene import read_vertices


def audit(ctx):
    def inventory():
        return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in ctx.out.rglob('*') if p.is_file()}
    before=inventory()
    with patch('industrial_swap_v2.core.local_background_optimizer.gsle',side_effect=AssertionError('No renderer allowed')), \
         patch('industrial_swap_v2.core.multiview_masks.prepare_multiview',side_effect=AssertionError('Only existing masks allowed')):
        background=optimize_background(ctx)
    training=background.parent; summary=read_json(training/'summary.json')
    folder=Path(summary['visibility_folder']); manifest=read_json(folder/'manifest.json')
    for view in manifest['views']:
        with np.load(folder/'cache'/f"{view['name']}.npz") as cache:
            assert np.array_equal(cache['old_mask'],cache['old_alpha']>manifest['config']['old_alpha_threshold'])
            assert np.array_equal(cache['new_mask'],cache['object_alpha']>manifest['config']['new_alpha_threshold'])
            assert np.array_equal(cache['environment_mask'],~(cache['old_mask']|cache['new_mask']))
            assert not (cache['background_train_mask']&~cache['environment_mask']).any()
            assert not (cache['visibility_weight']&cache['legal_foreground']).any()
    obj=ctx.stage('m3')/'object_world.ply'; fixed=ctx.stage('m4')/'object_fixed.ply'
    assert sha256(obj)==sha256(fixed)==summary['object_sha256']
    vertices=read_vertices(background); object_vertices=read_vertices(obj)
    final=read_vertices(ctx.stage('m4')/'scene_final.ply')
    assert final[:len(vertices)].tobytes()==vertices.tobytes()
    assert final[len(vertices):].tobytes()==object_vertices.tobytes()
    for name,digest in summary['source_hashes'].items(): assert sha256(manifest['sources'][name])==digest
    assert before==inventory(),'Existing results were modified'
    return {'run':str(ctx.out),'status':'read_only_audit_passed','camera_caches_checked':len(manifest['views']),
        'existing_files_unchanged':len(before),'completed_steps':summary['iterations'],
        'guarded_selected_step':summary['selected_step'],'device_count':len(object_vertices),
        'fixed_device_sha256':sha256(obj),'background_count':len(vertices),
        'scene_matches_optimized_background':True,'union_and_gradient_masks_valid':True,
        'qwen_calls':0,'trellis_calls':0,'optimization_steps_executed':0,'renders_executed':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--report')
    args=parser.parse_args(); ctx=Context(args.config)
    report=ctx.path(args.report) if args.report else None
    if report and (report.exists() or ctx.out in report.parents):
        raise ValueError('Use a NEW report path outside the original experiment')
    result=audit(ctx)
    if report:
        report.parent.mkdir(parents=True,exist_ok=True)
        with report.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
