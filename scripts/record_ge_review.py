"""Record actual GE outcome and the rejected input-adaptation diagnostic."""
import json,os
from pathlib import Path
root=Path(os.environ['ROOT'])
review={
 'method':'gaussianeditor',
 'run':'gaussianeditor_20261104T091151Z',
 'execution':'completed2500steps_exit0',
 'artifacts':'validated finite PLY,350 common RGB renders,44 held-out renders and350-frame video',
 'visual_result':'Target remains predominantly white; the requested red edit is weak/unsuccessful. Do not interpret the red semantic-overlay panel as edited RGB.',
 'inspected_actual_rgb':['frame_000061.png','frame_000091.png','frame_000151.png','frame_000161.png','frame_000281.png'],
 'segmentation':'The official traced mask at0.5 covers84-88% of the reviewed base in5 views; the model output is still not visibly red.',
 'limitations':'Single seed; industrial domain and small repeated objects challenge IP2P instruction grounding.',
 'additional_diagnostics':{
  'original_views':'outputs/prompt_diagnostic:4 equivalent English phrasings x2 seeds; largely unchanged target or red foreground table',
  'target_views':'outputs/prompt_diagnostic_target: calibrated target-centered views still produce wrong-object changes',
  'camera_check':'outputs/target_view_check/alignment.json: source-render vs resampled real image PSNR26.89-28.55dB in4 views',
  'decision':'Keep original common editing inputs and original English instruction for the main4-method baseline. Target-centered input adaptation is exploratory only and is not used in the main runs.'
 },
 'eligible_for_comparison':True,
 'quality_label':'execution_success_edit_target_not_achieved'
}
(root/'records/gaussianeditor.quality_review.json').write_text(json.dumps(review,indent=2))
manifest=root/'outputs/gaussianeditor/artifact_manifest.json'
data=json.loads(manifest.read_text())
data['status']='reviewed_actual_result_edit_target_not_achieved'
data['quality_review']=str(root/'records/gaussianeditor.quality_review.json')
manifest.write_text(json.dumps(data,indent=2))
run_path=Path(data['run_record'])
run_record=json.loads(run_path.read_text())
run_record.update(status='completed_reviewed',quality_label=review['quality_label'],quality_review=str(root/'records/gaussianeditor.quality_review.json'))
run_path.write_text(json.dumps(run_record,indent=2))
(root/'records/gaussianeditor.reviewed').write_text(review['quality_label'])
metric=root/'records/common_metrics.json'
if metric.exists(): (root/'records/gaussianeditor_attempt02_metrics.json').write_bytes(metric.read_bytes())
print(json.dumps(review,indent=2))
