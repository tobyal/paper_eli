import json,os,time
from pathlib import Path
root=Path(os.environ['ROOT'])
run=root/'records/runs/dge_20261104T094357Z/run.json'
for _ in range(30):
 record=json.loads(run.read_text())
 if record['status']!='running':break
 time.sleep(1)
assert record['status']!='running'
record.update(status='aborted_for_numerical_domain_fix',reason='Actual dot product1.0000000000000004 in signed camera-angle calculation caused NaN. Clamp to mathematical[-1,1]domain before arccos; no other algorithm changes.',excluded_from_final_timing_comparison=True)
for before,after in [(root/'outputs/dge',root/'outputs/dge_attempt01_angle_nan'),(root/'repos/DGE/edit_cache/industrial_red_dge',root/'repos/DGE/edit_cache/industrial_red_dge_attempt01_angle_nan')]:
 if before.exists():
  assert not after.exists()
  before.rename(after)
run.write_text(json.dumps(record,indent=2))
print(record['status'])
