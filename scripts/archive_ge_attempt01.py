import json
import os
from pathlib import Path
root=Path(os.environ['ROOT'])
run=root/'records/runs/gaussianeditor_20261104T090522Z/run.json'
record=json.loads(run.read_text())
assert record['status']!='running', 'Wait for terminated child process to exit'
record['status']='aborted_after_localization_quality_check'
record['reason']='Official texture example mask_thres0.8 selected452 Gaussians and covered only0.2-4.0% of reviewed housing in5 views. Replay of the same official semantic tracing at generic default0.5 covers84.4-87.8%; core algorithm unchanged.'
record['excluded_from_final_timing_comparison']=True
record['archived_output']=str(root/'outputs/gaussianeditor_attempt01_incomplete')
record['archived_cache']=str(root/'repos/GaussianEditor/edit_cache/industrial_red_gaussianeditor_attempt01_incomplete')
for before,after in [(root/'outputs/gaussianeditor',Path(record['archived_output'])),(root/'repos/GaussianEditor/edit_cache/industrial_red_gaussianeditor',Path(record['archived_cache']))]:
    if before.exists():
        assert not after.exists()
        before.rename(after)
run.write_text(json.dumps(record,indent=2))
adapt=root/'records/industrial_adaptation.json'
data=json.loads(adapt.read_text())
data['semantic_tracing_threshold']={'value':.5,'applies_to':['GaussianEditor','DGE','DreamCatalyst'],'reason':'High0.8 threshold rejected most target Gaussians due repeated-device views/occlusions; use GE and DreamCatalyst generic default0.5 for the common scene. DGE changes from its yaml0.8. Five diagnostic view coverage84.4-87.8%.','proof':'records/gaussianeditor_tracing_diagnostics.json'}
adapt.write_text(json.dumps(data,indent=2))
print(record['status'])
