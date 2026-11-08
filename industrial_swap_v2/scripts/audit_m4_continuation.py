"""Read-only verification after an M4 continuation; never launches training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..core.context import read_json,write_json,sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); root=args.run.resolve()
    launch=read_json(root/'launch.json'); exit_record=read_json(root/'process_exit.json')
    assert exit_record['returncode']==0,exit_record
    training=root/'m4/background_optimization'
    resume=read_json(training/'resume_verification.json'); final=read_json(root/'m4/validation/report.json')
    assert resume['start_step']==launch['start_step'] and final['rendered_step']==launch['target_step']
    logs=[json.loads(line) for line in (training/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(launch['start_step']+1,launch['target_step']+1))
    source=Path(launch['source_checkpoint']); initial=read_json(source.parent/'metrics.json')
    restored=read_json(training/'resume_metrics.json')
    max_diff=max(abs(a[k]-b[k]) for a,b in zip(initial['views'],restored['views'])
        for k in a if isinstance(a[k],(int,float)))
    assert max_diff<1e-6,max_diff
    source_checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    rng=np.random.default_rng(); rng.bit_generator.state=source_checkpoint['rng']
    order=list(source_checkpoint['remaining_view_order'])
    manifest=read_json(root/'m4/multiview_supervision/manifest.json')
    train=[v['name'] for v in manifest['views'] if v['split']=='train']; anchor=manifest['views'][0]['name']
    for row in logs:
        if not order: order=list(rng.permutation(len(train)))
        view=train[order.pop()]
        assert row['rgb_view']==view
        assert row['visibility_view']==(anchor if row['step']%4==0 else view)
    snapshots=[]
    for step in launch['save_steps']:
        folder=root/'m4/snapshots'/f'iteration_{step:06d}'
        saved=read_json(folder/'snapshot.json'); metrics=read_json(folder/'metrics.json')
        assert saved['step']==metrics['step']==step
        for name,digest in saved['files_sha256'].items(): assert sha256(folder/name)==digest,(step,name)
        for key,value in saved['checks'].items():
            if isinstance(value,bool): assert value,key
        assert saved['checks']['added_points']==saved['checks']['deleted_points']==0
        checkpoint=torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=False)
        assert checkpoint['step']==step
        assert all(int(state['step'].item())==step for state in checkpoint['optimizer']['state'].values())
        assert saved['object_sha256']==final['candidate_sha256']['object']
        snapshots.append({'step':step,'directory':str(folder),'guard_passed':saved['guard_passed'],
            'anchor_visibility':metrics['summary']['anchor']['visibility'],
            'validation_visibility':metrics['summary']['validation']['visibility'],
            'validation_environment_l1':metrics['summary']['validation']['environment_l1'],
            'validation_outside_union_mae':metrics['summary']['validation']['outside_union_mae'],
            'worst_outside_union_mae':max(v['outside_union_mae'] for v in metrics['views']),
            'worst_legal_foreground_mae':max(v['legal_foreground_mae'] for v in metrics['views']),
            'verified_files':len(saved['files_sha256'])})
    preserved=read_json(root/'previous_run_sha256.json')
    for name,digest in preserved.items(): assert sha256(name)==digest,name
    for name,digest in read_json(root/'source_preservation.json').items(): assert sha256(name)==digest,name
    checks=final['checks']
    assert checks['background_added']==checks['background_deleted']==checks['qwen_calls']==checks['trellis_calls']==0
    assert not checks['di_sds_executed']
    report={'status':'completed_and_verified','run':str(root),'start_step':launch['start_step'],
        'target_step':launch['target_step'],'executed_steps':len(logs),'source_checkpoint':str(source),
        'source_checkpoint_sha256':sha256(source),'resume_metric_max_absolute_difference':max_diff,
        'all_training_camera_sampling_verified':True,'previous_run_files_unchanged':len(preserved),
        'source_inputs_unchanged':True,'seconds':exit_record['seconds'],'snapshots':snapshots,
        'checks':checks,'selected_step':final['selected_step'],'candidate_guard_passed':final['candidate_guard_passed'],
        'actual_final_scene':str(root/'m4/snapshots/iteration_010000/scene.ply'),
        'guarded_final_scene':final['guarded_scene']}
    write_json(root/'run_verification.json',report)
    record={**launch,'status':'completed_and_verified','last_step':launch['target_step'],
        'verification':str(root/'run_verification.json'),'candidate_guard_passed':final['candidate_guard_passed']}
    write_json(Path(__file__).parents[1]/'records/m4_resume2000_to10000_active.json',record)
    lines=[f"M4 continuation {launch['start_step']} -> {launch['target_step']}",
        f"Executed steps: {len(logs)}; original experiment files preserved: {len(preserved)}",
        f"Checkpoint resume metric difference: {max_diff}; all camera sampling verified",
        'Fixed object; local background SH0 and opacity only; no Qwen/TRELLIS/DI-SDS.',
        'step | anchor visibility | validation visibility | validation environment MAE | guard']
    for s in snapshots:
        lines.append(f"{s['step']} | {s['anchor_visibility']:.6f} | {s['validation_visibility']:.6f} | {s['validation_outside_union_mae']:.6f} | {s['guard_passed']}")
    lines += [f"Actual final iteration: {report['actual_final_scene']}",
        f"Guarded final selection step: {final['selected_step']}",
        'Inspect snapshots for actual iterations; scene_final.ply uses the guarded selection.']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
