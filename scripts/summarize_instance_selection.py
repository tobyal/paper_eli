"""Describe candidate-selection coverage without treating it as edit success."""
import json,os
from pathlib import Path
root=Path(os.environ['ROOT']);summary={}
for method in ['gaussianeditor','dge','dreamcatalyst','editsplat']:
 runs=sorted((root/'records/runs').glob(method+'_*'))
 if not runs:continue
 path=runs[-1]/'instance_selection.jsonl'
 if not path.exists():continue
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 summary[method]={'run':runs[-1].name,'calls':len(rows),'nonempty_selected_mask':sum(r['selected_pixels']>0 for r in rows),'no_candidate':sum(r['candidate_count']==0 for r in rows),'offscreen_projected_roi':sum(not r['projected_roi'] for r in rows),'note':'Calls may include repeat/update views. This measures localization availability, not editing quality.'}
(root/'records/instance_selection_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
