"""Register an already completed, verified isolated run; does not queue models."""
import argparse
from pathlib import Path
import shutil
import yaml
from . import store as s, service as v
from ..core.context import read_json, write_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-experiment',default='9b0eeee5840d453e8f21e1763544a6cb')
    args=parser.parse_args(); cfg=yaml.safe_load(args.config.read_text())
    root=(s.BASE/cfg['project']['output_dir']).resolve(); root.relative_to(s.BASE.resolve())
    completed=read_json(root/'validation/completed.json'); assert completed['all_checks_passed']
    assert s.sha(root/'m4/scene_final.ply')==completed['final_scene_sha256']
    record=s.BASE/'records/multiview_web_validation.json'
    if record.exists():
        existing=read_json(record)
        if existing['root']==str(root) and all(v.verify_outputs(v.run(rid)) for rid in existing['runs'].values()):
            print(existing); return
        raise ValueError('A different or changed Web result is already registered')
    source=v.experiment(args.source_experiment); source_heads=s.unpack(source['heads'])
    eid=v.create('多视角局部场景优化 · 原始设备位置保持')
    with s.transaction() as conn:
        s.sql(conn,'UPDATE experiments SET config=:cfg,heads=:heads WHERE id=:id',id=eid,cfg=s.dump(cfg),
              heads=s.dump({k:r for k,r in source_heads.items() if k in ('m1','m2','materials')}))
    # Make comparison files reachable through the standard per-run file endpoint.
    shutil.copytree(root/'validation',root/'m4/validation',dirs_exist_ok=True)
    runs={}; inputs={k:source_heads[k] for k in ('m1','m2')}
    for stage in ('m3','m4'):
        folder=s.DATA/'experiments'/eid/'runs'/s.uid(); folder.mkdir(parents=True)
        rid=folder.name; config=folder/'config.yaml'; config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        log=root/('m4/training.jsonl' if stage=='m4' else 'm3/multiview_visibility/occlusion_candidates.json')
        with s.transaction() as conn:
            s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,
              config_path,folder,log_path,created_at,request_id) VALUES(:id,:e,:stage,'historical','running',:sig,:inputs,:params,:cfg,:folder,:log,:t,:id)''',
              id=rid,e=eid,stage=stage,sig=s.digest({'validated_multiview':completed,'stage':stage}),inputs=s.dump(inputs),
              params=s.dump({'imported_verified_result':True,'iterations':2000} if stage=='m4' else {'realign':False,'object_occlusion_filter':{'enabled':False}}),
              cfg=str(config),folder=str(folder),log=str(log),t=s.now())
            s.sql(conn,'''INSERT INTO generation_records(id,experiment_id,run_id,model,prompt,input_assets,status,created_at)
                VALUES(:id,:e,:r,'verified multiview refinement',:prompt,:inputs,'running',:t)''',id=s.uid(),e=eid,r=rid,
                prompt=cfg['m2']['replacement_text'],inputs=s.dump(inputs),t=s.now())
        v.finish(rid,root/stage)
        runs[stage]=rid; inputs[stage]=rid
    assert s.unpack(v.experiment(args.source_experiment)['heads'])==source_heads
    result={'experiment':eid,'runs':runs,'root':str(root),'source_experiment':args.source_experiment,
        'source_heads_unchanged':True,'no_qwen_or_trellis_calls':True,'completed':completed}
    write_json(record,result); print(result)


if __name__=='__main__': main()
