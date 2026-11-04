"""Compact status snapshot; completed markers never imply quality approval."""
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
root=Path(os.environ.get('ROOT','/mnt/sda/ubuntu/lkx/papers/paper_eli'))
status={'time_utc':datetime.now(timezone.utc).isoformat()}
status['completed_markers']=[p.name for p in (root/'records').glob('*.completed')]
log=root/'logs/01b_complete_matching.log'
if log.exists():
    text=log.read_text(errors='replace')
    registered=re.findall(r'Registering image #\d+ \((\d+)\)',text)
    status['latest_sfm_registered']=int(registered[-1]) if registered else 0
if (root/'records/sfm_summary.json').exists():
    status['sfm']=json.loads((root/'records/sfm_summary.json').read_text())
status['reconstruction_checkpoints']=[str(p.relative_to(root)) for p in (root/'outputs/reconstruction').glob('point_cloud/*/point_cloud.ply')]
status['runs']=[json.loads(p.read_text()) for p in (root/'records/runs').glob('*/run.json')]
status['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
(root/'records/current_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
