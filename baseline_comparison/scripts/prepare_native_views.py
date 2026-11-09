"""Select target-observing original training cameras; no mask substitution."""
import json,shutil
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
spec=json.loads((BASE/'prompts.json').read_text());source=ROOT/'data/edit_train'
lines=(source/'sparse/0/images.txt').read_text().splitlines()
poses={line.split()[-1]:line for line in lines if line and not line.startswith('#')}
for case in spec['cases']:
    records=json.loads((Path(case['ours_source'])/'m4/multiview_supervision/manifest.json').read_text())['views']
    observed=[]
    for row in records:
        name=Path(row.get('original_image') or '').name
        if name not in poses:continue
        counts=row['region_pixel_counts'];area=counts['old_only']+counts['overlap']
        observed.append((area,name))
    observed.sort(reverse=True)
    selected=list(dict.fromkeys([case['reference_image']]+[n for a,n in observed if a>=100]))[:20]
    assert len(selected)>=12,(case['case'],len(selected))
    dest=BASE/'inputs'/case['case']/'native_views';(dest/'sparse/0').mkdir(parents=True,exist_ok=True);(dest/'images').mkdir(exist_ok=True)
    shutil.copyfile(source/'sparse/0/cameras.txt',dest/'sparse/0/cameras.txt')
    (dest/'sparse/0/images.txt').write_text(''.join(poses[n]+'\n\n' for n in sorted(selected)))
    for name in selected:
        p=dest/'images'/name
        if not p.exists():p.symlink_to(source/'images'/name)
    cloud=dest/'sparse/0/points3D.ply'
    if not cloud.exists():cloud.symlink_to(source/'sparse/0/points3D.ply')
    case['native_view_source']=str(dest)
    (dest/'selection.json').write_text(json.dumps({'policy':'Up to 20 original training cameras with the largest visible OLD-device area; original M1 reference included. Native semantic segmentation is rerun, no old masks enter its segmentation or training.','source':str(source),'selected':selected,'ranking':observed},indent=2))
    for method in spec['methods']:(BASE/method/case['case']/'task.json').write_text(json.dumps(case,indent=2))
(BASE/'prompts.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False))
print('Prepared per-case original camera subsets',[(c['case'],len(json.loads((Path(c['native_view_source'])/'selection.json').read_text())['selected'])) for c in spec['cases']])
