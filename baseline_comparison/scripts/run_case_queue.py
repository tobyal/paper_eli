"""Sequential case queue; wait for existing case00, never claim queued success."""
import argparse,subprocess,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('method',choices=['dge','gsle','ge']);p.add_argument('--gpu',required=True);p.add_argument('--through');p.add_argument('--wait-case00',action='store_true');a=p.parse_args()
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
if a.wait_case00:
 status=BASE/('gaussianeditor' if a.method=='ge' else a.method)/'case00/records/status.json'
 while json.loads(status.read_text())['status']=='running':time.sleep(15)
for case in ['case02','case04','case07','case10']:
 cmd=['python3',str(BASE/'scripts'/('run_'+a.method+'.py')),case,'--gpu',a.gpu]
 if a.through:cmd+=['--through',a.through]
 print('START',case,flush=True);rc=subprocess.run(cmd,cwd=ROOT).returncode
 print('FINISHED',case,'exit',rc,flush=True)
