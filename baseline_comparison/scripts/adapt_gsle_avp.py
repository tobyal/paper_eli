"""Permit explicitly recorded orbit geometry changes, retaining AVP scoring."""
from pathlib import Path
import json,shutil,difflib,argparse
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--radius',type=float,default=.8);p.add_argument('--elevation',type=float,default=15);p.add_argument('--min-visibility',type=float,default=.9);a=p.parse_args()
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';code=BASE/'vendor/gsle/avp.py'
t=code.read_text()
if 'radius_multiplier' not in t:
 u=t.replace('    records = []','    radius *= float(cfg["avp"].get("radius_multiplier", 1.0))\n    elevation = float(np.clip(elevation + cfg["avp"].get("elevation_offset_degrees", 0.0), 10, 65))\n    records = []',1)
 code.write_text(u);(BASE/'records/gsle_avp_orbit_parameters.patch').write_text(''.join(difflib.unified_diff(t.splitlines(True),u.splitlines(True),fromfile='before/avp.py',tofile='after/avp.py')))
out=BASE/'gsle'/a.case;cfg=json.loads((out/'config.json').read_text())
tag=f'avp_radius_{cfg["avp"].get("radius_multiplier",1)}_elevation_{cfg["avp"].get("elevation_offset_degrees",0)}'
dest=out/'attempts'/tag;dest.mkdir(parents=True)
for f in ['avp','logs/avp.log','config.json']:
 src=out/f
 if src.exists():
  if f=='config.json':shutil.copy2(src,dest/src.name)
  else:shutil.move(str(src),str(dest/src.name))
cfg['avp'].update(radius_multiplier=a.radius,elevation_offset_degrees=a.elevation,min_visibility=a.min_visibility)
(out/'config.json').write_text(json.dumps(cfg,indent=2))
(out/'records'/('adaptation_'+tag+'.json')).write_text(json.dumps({'reason':'Original automatic orbit had no candidate satisfying the existing industrial visibility safety gate. Preserve native 100 azimuths, eight image rotations, brightness criterion and 0.75 scene coverage. Explicit camera/gate adaptation below; not a learned model change.','radius_multiplier':a.radius,'elevation_offset_degrees':a.elevation,'minimum_visibility':a.min_visibility,'previous_attempt':str(dest)},indent=2))
print(a.case,cfg['avp'])
