"""Compact read-only progress for the independent weak DI-SDS experiment."""
import json
from pathlib import Path

root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison/gsle_weak_disds')
for name in ['pilot_case04_500','case00','case02','case04','case07','case10']:
    p=root/name
    if not (p/'records/status.json').exists():
        print(name,'pending');continue
    s=json.loads((p/'records/status.json').read_text())
    row=dict(case=name,status=s['status'],stage=s['active_stage'],gpu=s['gpu'])
    log=p/'refine/training.jsonl'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-20000))
            lines=f.read().splitlines()
        for line in reversed(lines):
            try:r=json.loads(line)
            except (ValueError,UnicodeError):continue
            row.update(step=r['step'],rgb_mse=round(r['rgb_mse'],7),mask_mse=round(r['mask_mse'],7))
            break
    print(json.dumps(row),flush=True)
