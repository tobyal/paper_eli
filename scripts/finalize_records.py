"""Verify delivered artifacts and mark the actual experiment, not edit success."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

root=Path(os.environ['ROOT'])
methods=['gaussianeditor','dge','dreamcatalyst','editsplat']
metrics=json.loads((root/'records/common_metrics.json').read_text())
comparison=json.loads((root/'records/final_comparison.json').read_text())
assert len(comparison['rows'])==4
checks={}
for method in methods:
    dest=root/'outputs'/method
    artifact=json.loads((dest/'artifact_manifest.json').read_text())
    run=json.loads(Path(artifact['run_record']).read_text())
    quality=json.loads((root/f'records/{method}.quality_review.json').read_text())
    model=json.loads((dest/'model_check.json').read_text())
    assert run['status']=='completed_reviewed' and run['exit_code']==0
    assert quality['eligible_for_comparison']
    assert model['all_parameters_finite'] and model['all_quaternions_nonzero']
    render=dest/'common_render'
    wide=len(list((render/'wide').glob('*.png')))
    test=len(list((render/'test_512').glob('*.png')))
    assert wide==350 and test==44, (method,wide,test)
    assert len(metrics['methods'][method]['views'])==44
    video=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(render/'multiview.mp4')],text=True))
    stream=video['streams'][0]
    assert int(stream['nb_frames'])==350
    assert (stream['width'],stream['height'])==(960,540)
    assert abs(float(video['format']['duration'])-350/12)<.1
    checks[method]={'execution':'completed_reviewed','quality_label':quality['quality_label'],'wide_renders':wide,'square_test_renders':test,'video':video,'ply_sha256':model['sha256']}
grid=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(root/'outputs/comparison/four_methods_multiview.mp4')],text=True))
assert int(grid['streams'][0]['nb_frames'])==350
checks['comparison_video']=grid
checks['verified_utc']=datetime.now(timezone.utc).isoformat()
(root/'records/final_artifact_verification.json').write_text(json.dumps(checks,indent=2))
experiment=json.loads((root/'records/experiment.json').read_text())
experiment['evaluation'].update(status='completed_actual_outputs_visually_reviewed',metrics_record='records/common_metrics.json',comparison_record='records/final_comparison.json',artifact_verification='records/final_artifact_verification.json',quality_labels={m:checks[m]['quality_label'] for m in methods})
(root/'records/experiment.json').write_text(json.dumps(experiment,ensure_ascii=False,indent=2))
print(json.dumps(checks,indent=2))
