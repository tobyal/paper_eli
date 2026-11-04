import argparse,hashlib,json,os
from pathlib import Path
root=Path(os.environ['ROOT'])
p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--result',required=True);p.add_argument('--label',required=True)
p.add_argument('--result-zh')
a=p.parse_args();dest=root/'outputs'/a.method
manifest=json.loads((dest/'artifact_manifest.json').read_text())
run=json.loads(Path(manifest['run_record']).read_text())
assert run['exit_code']==0
frames=[f'frame_{i:06d}.png' for i in [61,91,151,161,281]]
review={'method':a.method,'run':Path(manifest['run_record']).parent.name,
 'execution':'official_entrypoint_exit0','visual_result':a.result,'quality_label':a.label,
 'inspected_actual_rgb':frames,'inspection_sheet':str(dest/'quality_preview.png'),
 'inspection_sheet_sha256':hashlib.sha256((dest/'quality_preview.png').read_bytes()).hexdigest(),
 'artifacts':'finite PLY,350 common RGB renders,44 held-out square renders,350-frame multiview video',
 'eligible_for_comparison':True}
if a.result_zh: review['visual_result_zh']=a.result_zh
(root/f'records/{a.method}.quality_review.json').write_text(json.dumps(review,indent=2))
manifest.update(status='reviewed_actual_result',quality_review=str(root/f'records/{a.method}.quality_review.json'))
(dest/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
(root/f'records/{a.method}.reviewed').write_text(a.label)
run.update(status='completed_reviewed',quality_label=a.label,quality_review=str(root/f'records/{a.method}.quality_review.json'))
Path(manifest['run_record']).write_text(json.dumps(run,indent=2))
print(json.dumps(review,indent=2))
