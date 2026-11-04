"""Validate and render a finished official run; visual approval remains separate."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
root=Path(os.environ['ROOT'])
parser=argparse.ArgumentParser()
parser.add_argument('method',choices=['gaussianeditor','dge','dreamcatalyst','editsplat'])
parser.add_argument('--run',type=Path)
args=parser.parse_args()
method=args.method
run_dir=args.run or sorted((root/'records/runs').glob(method+'_*'))[-1]
deadline=time.monotonic()+8*3600
while True:
    run=json.loads((run_dir/'run.json').read_text())
    if run['status']!='running': break
    if time.monotonic()>deadline: raise TimeoutError('Official entrypoint still running after8 hours')
    time.sleep(10)
assert run['status']=='entrypoint_completed_needs_output_review',run
output_name=run.get('output_name',method)
destination=root/'outputs'/output_name
if method=='editsplat':
    candidates=list((destination/'point_cloud').glob('iteration_*/point_cloud.ply'))
    ply=max(candidates,key=lambda p:int(p.parent.name.split('_')[-1]))
else:
    candidates=list(destination.glob('**/save/last.ply'))
    assert len(candidates)==1,candidates
    ply=candidates[0]
python=str(root/'envs/reconstruction/bin/python')
subprocess.run([python,str(root/'scripts/inspect_model.py'),str(ply),'--output',str(destination/'model_check.json')],check=True)
start=time.monotonic()
subprocess.run([python,str(root/'scripts/render_common.py'),'--ply',str(ply),'--name',output_name],check=True)
elapsed=time.monotonic()-start
render=destination/'common_render'
assert len(list((render/'wide').glob('*.png')))==350
assert len(list((render/'test_512').glob('*.png')))==44
video=render/'multiview.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames:format=duration,size','-of','json',str(video)],text=True))
assert video.stat().st_size>10000
manifest={'method':method,'run_record':str(run_dir/'run.json'),'ply':str(ply),'model_check':str(destination/'model_check.json'),'common_render_dir':str(render),'multiview_video':str(video),'common_render_seconds':elapsed,'video_probe':probe,'status':'real_artifacts_generated_awaiting_visual_review','created_utc':datetime.now(timezone.utc).isoformat()}
(destination/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
(root/f'records/{output_name}.rendering.completed').write_text(datetime.now(timezone.utc).isoformat())
print(json.dumps(manifest,indent=2),flush=True)
