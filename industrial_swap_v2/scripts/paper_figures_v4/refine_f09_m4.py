"""Run the official dual-mask M4 on immutable, already placed F09 assets.

No Qwen/TRELLIS, registration, floor completion or device-parameter update.
Outputs are separate from the previous placement-only experiment.
"""
import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
sys.path.insert(0, str(ROOT))
from industrial_swap_v2.core.context import Context, read_json, write_json, sha256
from industrial_swap_v2.core.workflow import validate_workflow
from industrial_swap_v2.core.local_background_optimizer import optimize_background
from industrial_swap_v2.core.m4_outputs import export_m4
from industrial_swap_v2.core.scene import compose

BASE = ROOT / 'industrial_swap_v2'
F09 = BASE / 'outputs/paper_figures_v4/F09_text_control'
DEST = F09 / 'm4_2000_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['B_scara', 'C_cnc', 'D_conveyor'])
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    source = F09 / 'variants' / args.variant
    dest = DEST / 'variants' / args.variant
    if (dest / 'completed.json').exists():
        saved = read_json(dest / 'completed.json')
        assert sha256(dest / 'm4/scene_candidate_step2000.ply') == saved['candidate_scene_sha256']
        print('REUSED COMPLETED', args.variant, flush=True)
        return
    if dest.exists():
        raise FileExistsError('Partial run exists; inspect its log instead of restarting: ' + str(dest))
    dest.mkdir(parents=True)
    old = Path(read_json(ROOT / 'baseline_comparison/prompts.json')['cases'][0]['ours_source'])
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    robot_cfg = yaml.safe_load((old / 'config.yaml').read_text())
    cfg['project'].update(output_dir=str(dest.relative_to(BASE)),
                          records_dir=str((dest / 'records').relative_to(BASE)),
                          logs_dir=str((dest / 'logs').relative_to(BASE)), gpu=args.gpu)
    cfg['m4']['local_background'] = copy.deepcopy(robot_cfg['m4']['local_background'])
    cfg['m4']['local_background'].update(enabled=True, iterations=2000)
    cfg['m4']['multiview_supervision'] = copy.deepcopy(robot_cfg['m4']['multiview_supervision'])
    cfg['m4']['save_every'] = 500
    cfg['m4'].pop('resume', None)
    validate_workflow(cfg)
    for stage in ('m1', 'm2', 'm3'):
        (dest / stage).symlink_to((source / stage).resolve(), target_is_directory=True)
    (dest / 'm4').mkdir()
    for name in ('object_world.ply', 'transform.json'):
        shutil.copy2(source / 'm4' / name, dest / 'm4' / name)
    config = dest / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    ctx = Context(config)
    placed = next(r for r in read_json(F09 / 'placement_results.json')['variants'] if r['variant'] == args.variant)
    assert sha256(ctx.stage('m1') / 'background_gaussians.ply') == placed['background_sha256']
    assert sha256(ctx.stage('m4') / 'object_world.ply') == placed['object_sha256']
    preservation = {str(source / 'm4' / n): sha256(source / 'm4' / n)
                    for n in ('object_world.ply', 'transform.json')}
    preservation.update({str(source / 'scene_final.ply'): placed['scene_sha256'],
                         str(ctx.stage('m1') / 'background_gaussians.ply'): placed['background_sha256']})
    write_json(dest / 'source_preservation.json', preservation)
    write_json(ctx.stage('m4') / 'fixed_placement_reuse.json', {
        'source': str(source), 'registration_reexecuted': False,
        'object_sha256': placed['object_sha256'], 'transform': placed['transform'],
        'policy': 'Preserve exact previous placement; only official dual-mask local background optimization.'})
    compose(ctx.stage('m1') / 'background_gaussians.ply', ctx.stage('m4') / 'object_world.ply',
            ctx.stage('m4') / 'scene_initial.ply', ctx.stage('m4') / 'initial_composition.json')
    torch.cuda.set_per_process_memory_fraction(.60)
    start = time.time()
    optimize_background(ctx)
    export_m4(ctx)
    report = read_json(ctx.stage('m4') / 'summary.json')
    assert report['checks']['training_steps'] == 2000
    assert sha256(ctx.stage('m4') / 'object_world.ply') == placed['object_sha256']
    assert sha256(ctx.stage('m4') / 'transform.json') == preservation[str(source / 'm4/transform.json')]
    write_json(dest / 'completed.json', {
        'variant': args.variant, 'steps': 2000, 'seconds_including_supervision_export': time.time() - start,
        'candidate_scene_sha256': report['candidate_sha256']['scene'],
        'candidate_guard_passed': report['candidate_guard_passed'], 'guarded_selected_step': report['selected_step'],
        'new_object_masks_per_view': True, 'reused_robot_masks': False,
        'qwen_calls': 0, 'trellis_calls': 0, 'registration_reexecuted': False,
        'object_geometry_appearance_and_pose_unchanged': True})
    print('F09 M4 COMPLETE', args.variant, flush=True)


if __name__ == '__main__':
    main()
