"""One recovery pass for recorded long-prompt failures after the active queue."""
import time,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
while Path('/proc/1275285').exists():time.sleep(15)
for case in ['case04','case07','case10']:
 out=BASE/'dge'/case;status=out/'records/status.json'
 if not status.exists() or json.loads(status.read_text())['status']!='failed':continue
 text=(out/'logs/train.log').read_text(errors='replace')
 if 'expected sequence of length' not in text:print('Unrelated failure requires inspection',case,flush=True);continue
 dest=out/'attempts/04_long_prompt_without_truncation';dest.mkdir(parents=True)
 for name in ['official','edit_cache','logs/train.log','records/status.json','records/instance_selection.jsonl']:
  p=out/name
  if p.exists():shutil.move(str(p),str(dest/p.name))
 rc=subprocess.run(['python3',str(BASE/'scripts/run_dge.py'),case,'--gpu','0'],cwd=ROOT).returncode
 print(case,'prompt-compatibility retry',rc,flush=True)
