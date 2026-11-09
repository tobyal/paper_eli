"""Resume completed Delete after a transient GPU-memory collision, no regeneration."""
import json,time,subprocess,shutil
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
while True:
 d=json.loads((BASE/'gaussianeditor/case10/records/status.json').read_text())
 if d['status'] in ['training_complete','failed']:break
 time.sleep(15)
status=BASE/'gaussianeditor/case04/records/status.json'
shutil.copy2(status,BASE/'records/ge_case04_shared_gpu_oom.json')
raise SystemExit(subprocess.call(['python3',str(BASE/'scripts/run_ge.py'),'case04','--gpu','0'],cwd=ROOT))
