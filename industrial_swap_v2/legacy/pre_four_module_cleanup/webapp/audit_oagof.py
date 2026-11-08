"""Audit the real Web reuse path and optionally exercise exact stage caching."""
import argparse
import json
from pathlib import Path
from . import store as s, service as v
from ..core.context import Context
from ..core.m3_composition import effective_background


def main():
    p=argparse.ArgumentParser(); p.add_argument('--submit-cache',action='store_true'); args=p.parse_args()
    path=s.BASE/'records/oagof_web_validation.json'; record=json.loads(path.read_text())
    current=v.run(record['run']); assert current['status']=='completed',current['status']
    directory=v.stage_dir(record['run']); report=json.loads((directory/'occlusion_filter/occlusion_report.json').read_text())
    assert s.unpack(v.experiment(record['source_experiment'])['heads'])==record['source_heads']
    assert all(v.verify_outputs(v.run(rid)) for rid in record['source_heads'].values())
    assert s.sha(directory/'object_world.ply')==record['world_sha256']
    assert json.loads((directory/'reuse.json').read_text())['trellis_called'] is False
    assert not (directory/'trellis_generation.json').exists()
    assert not list(directory.rglob('floor_patch.ply'))
    standalone=s.BASE/'outputs/oagof/current_m3_v2'
    for name in ('background_filtered.ply','removed_occluders.ply','scene_filtered.ply'):
        assert s.sha(directory/'occlusion_filter'/name)==s.sha(standalone/name),name
    assert s.sha(directory/'scene_coarse.ply')==s.sha(standalone/'scene_filtered.ply')
    preserved=json.loads((standalone/'preservation.json').read_text())
    assert all(s.sha(Path(name))==digest for name,digest in preserved['sha256'].items())
    assert effective_background(Context(current['config_path']))==(directory/'occlusion_filter/background_filtered.ply').resolve()
    validation=json.loads((standalone/'validation/validation.json').read_text()); assert validation['status']=='passed'
    record.update(status='verified',original_heads_unchanged=True,original_outputs_sha256_verified=len(preserved['sha256']),
        object_sha256_unchanged=True,no_qwen_or_trellis_calls=True,new_gaussians=0,
        web_matches_independently_validated_ply=True,held_out_validation=str(standalone/'validation/validation.json'),
        removed_count=report['removed_count'],unresolved_count=report['unresolved_candidate_count'],
        output_directory=str(directory),downstream_background_verified=True)
    old=s.BASE/'records/oagof_web_validation_v1.json'
    if old.exists():
        previous=json.loads(old.read_text())
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET name=:name WHERE id=:id',
                name='OAGOF v1 diagnostic - held-out guard failed',id=previous['experiment'])
    if args.submit_cache and not record.get('cache_run'):
        experiment=v.create('OAGOF cache validation',copy_from=record['experiment'])
        identifier=v.submit(experiment,'m3','reuse',s.uid(),parameters=s.unpack(current['parameters']))
        record.update(cache_experiment=experiment,cache_run=identifier)
    if record.get('cache_run'):
        cached=v.run(record['cache_run']); record['cache_status']=cached['status']
        if cached['status']=='completed':
            assert cached['reused_from']==record['run'],cached['reused_from']
            record['exact_stage_cache_verified']=True
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
