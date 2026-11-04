"""Read-only compact status for long-running official jobs."""
import json,os,re
from pathlib import Path
root=Path(os.environ['ROOT'])
for method in ['gaussianeditor','dge','dreamcatalyst','editsplat']:
 runs=sorted((root/'records/runs').glob(method+'_*'))
 if not runs: continue
 run=json.loads((runs[-1]/'run.json').read_text())
 log=Path(run['log']).read_text(errors='replace')
 lines=[re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',s).strip() for s in re.split(r'[\r\n]',log)]
 lines=[s for s in lines if s]
 print(json.dumps({'method':method,'run':runs[-1].name,'status':run['status'],'wall_seconds':run.get('wall_seconds'),'latest':lines[-4:]},ensure_ascii=False))
