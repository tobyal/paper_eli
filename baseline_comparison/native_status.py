"""Compact, read-only status for the ten independent native experiments."""
from pathlib import Path
import datetime,json,re

BASE=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison')


def tail(path,limit=2400):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));return f.read().decode(errors='replace')


print(datetime.datetime.now().isoformat())
for case in ['case00','case02','case04','case07','case10']:
    row={'case':case}
    for method in ['dge','gsle']:
        p=BASE/(method+'_native')/case;status=p/'records/status.json'
        if not status.exists():row[method]={'status':'prepared' if p.exists() else 'not_run'};continue
        state=json.loads(status.read_text());item={'status':state['status']}
        if method=='gsle':
            stage=state.get('active_stage');item['stage']=stage
            if stage in ['coarse','refine']:
                lines=tail(p/stage/'training.jsonl',65536).splitlines()
                if lines:
                    last=json.loads(lines[-1]);item.update(step=last['step'],points=last['points'])
        elif state['status']=='running':
            log=tail(p/'logs/train.log')
            steps=re.findall(r'Epoch.*?([0-9]+)it',log)
            item['phase']='fitting' if steps else ('editing_or_export' if (p/'target_indices.npy').exists() else 'localization')
            if steps:item['step']=int(steps[-1])
        verification=p/'records/verification.json'
        if verification.exists():item['verification']=json.loads(verification.read_text())['status']
        row[method]=item
    print(json.dumps(row))
