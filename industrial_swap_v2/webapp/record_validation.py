"""Summarize only observed database/output evidence for delivery."""
import json
import re
import subprocess
import sys
from pathlib import Path
from . import store as s
from . import service as v


def main():
    records=Path(__file__).parent/'records'; records.mkdir(exist_ok=True)
    live_path=s.DATA/'validation/live_run.json'; routes_path=s.DATA/'validation/asset_routes.json'
    live=json.loads(live_path.read_text()); routes=json.loads(routes_path.read_text()) if routes_path.exists() else {}
    values=[]
    identifiers=set(live.get('runs',{}).values())|{row['run'] for row in routes.values() if isinstance(row,dict) and 'run' in row}
    ui=s.rows("SELECT r.id FROM stage_runs r JOIN experiments e ON r.experiment_id=e.id WHERE e.name='Web UI 验证 · RGBA 素材复用'")
    identifiers.update(r['id'] for r in ui)
    for rid in sorted(identifiers):
        row=v.run(rid); out=s.unpack(row['outputs'])
        values.append({k:row[k] for k in ('id','experiment_id','stage','action','status','started_at','finished_at','reused_from')})
        values[-1]['verified']=v.verify_outputs(row) if row['status']=='completed' else False
        values[-1]['output_directory']=out.get('directory')
        values[-1]['assets']=out.get('assets',{})
        generation=s.one('SELECT seconds,external_api,metadata FROM generation_records WHERE run_id=:id',id=rid)
        values[-1]['execution']=dict(generation) if generation else {}
        if generation: values[-1]['execution']['metadata']=s.unpack(generation['metadata'])
        if row['stage']=='m4' and row['status']=='completed':
            values[-1]['refinement_summary']=json.loads((Path(out['directory'])/'summary.json').read_text())
    historical=all(s.sha(path)==checksum for path,checksum in live['baseline_hashes'].items())
    tests={}
    for name in ('web','cli'):
        log=records/('tests_'+name+'.log')
        if log.exists():
            content=log.read_text(); count=re.search(r'Ran (\d+) tests',content)
            tests[name]={'count':int(count.group(1)) if count else None,'passed':content.rstrip().endswith('OK')}
    source_hashes={str(p.relative_to(s.ROOT)):s.sha(p) for p in Path(__file__).parent.rglob('*')
        if p.is_file() and p.suffix in ('.py','.js','.css','.html','.ps1','.sh') and not any(k in p.parts for k in ('data','records','__pycache__'))}
    for name in ('core/context.py','core/qwen_client.py','modules/m2_image_generation.py','tests/test_qwen_once.py'):
        path=s.BASE/name; source_hashes[str(path.relative_to(s.ROOT))]=s.sha(path)
    qwen_claims=len(list((s.DATA/'experiments').glob('*/runs/*/records/qwen_generation_once.lock')))
    complete=(all(r['status']=='completed' and r['verified'] for r in values) and historical and qwen_claims==0
        and live.get('stage_cache_verified') and live.get('model_cache_verified') and routes.get('status')=='complete'
        and len(tests)==2 and all(t['passed'] for t in tests.values()))
    report={'checked_at':s.now(),'status':'complete' if complete else 'in_progress',
        'live_validation':live,'asset_routes':routes,'runs':values,'historical_artifacts_unchanged':historical,
        'web_qwen_submission_locks':qwen_claims,
        'tests':tests,'source_sha256':source_hashes,
        'python':sys.version,'database_tables':['experiments','assets','stage_runs','generation_records','experiment_assets','asset_events'],
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip()}
    (records/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (records/'webapp.pip.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    print(json.dumps({'status':report['status'],'runs':len(values),'historical_unchanged':historical,'qwen_calls':report['web_qwen_submission_locks']}))


if __name__=='__main__': main()
