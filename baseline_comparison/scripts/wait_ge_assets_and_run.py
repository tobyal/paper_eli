"""Wait for verified native assets, then run five official Delete/Add attempts."""
from pathlib import Path
import time,subprocess,json
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
while True:
 sd=BASE/'records/gaussianeditor_sd15.json';dpt=BASE/'records/gaussianeditor_recovered_models.json'
 if sd.exists() and dpt.exists() and 'omnidata' in json.loads(dpt.read_text()):break
 time.sleep(15)
for case in ['case00','case02','case04','case07','case10']:
 print('START',case,flush=True)
 rc=subprocess.run(['python3',str(BASE/'scripts/run_ge.py'),case,'--gpu','1'],cwd=ROOT).returncode
 print('FINISHED',case,rc,flush=True)
