"""Paper AVP with predeclared geometry-only orbit retries, never Ours views."""
import json,os,subprocess,sys,shutil
from pathlib import Path

root=Path(os.environ['ROOT']);out=Path(os.environ['GSLE_OUTPUT']);config_path=Path(os.environ['GSLE_CONFIG'])
cfg=json.loads(config_path.read_text());attempts=[]
# First valid orbit wins; brightness scores never choose the orbit itself.
for i,(radius,elevation) in enumerate([(1.0,0),(1.3,15),(1.6,30),(2.0,30)]):
    active=json.loads(json.dumps(cfg));active['avp'].update(radius_multiplier=radius,elevation_offset_degrees=elevation)
    path=out/'records'/f'avp_config_{i}.json';path.write_text(json.dumps(active,indent=2));env=os.environ.copy();env['GSLE_CONFIG']=str(path)
    with (out/'logs'/f'avp_orbit_{i}.log').open('w') as log:
        rc=subprocess.call([sys.executable,str(root/'baseline_comparison/vendor/gsle_native/avp.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    attempts.append(dict(attempt=i,radius_multiplier=radius,elevation_offset_degrees=elevation,exit_code=rc))
    (out/'records/avp_orbit_attempts.json').write_text(json.dumps(dict(reason='Predeclared orbit changes only if occlusion/scene-coverage constraints reject every view',attempts=attempts),indent=2))
    if rc==0:break
    candidates=out/'avp/candidates.json'
    if not candidates.exists() or any(r['eligible'] for r in json.loads(candidates.read_text())):raise SystemExit(rc)
    (out/'avp').rename(out/f'avp_rejected_orbit_{i}')
else:raise RuntimeError('No eligible AVP orbit; failed geometry retained without external anchor fallback')
