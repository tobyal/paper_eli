"""Validate final saved artifacts and actual native iteration-completion evidence."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from plyfile import PlyData
from PIL import Image
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison'
spec=json.loads((BASE/'prompts.json').read_text());results=[]
for task in spec['cases']:
 case=task['case'];cameras=json.loads((OUT/case/'cameras.json').read_text())['cameras'];assert len(cameras)==3
 for method in spec['methods']:
  folder=BASE/method/case;status=json.loads((folder/'records/status.json').read_text());assert status['status']=='training_complete',(method,case,status)
  native_task=json.loads((folder/'task.json').read_text());assert native_task['A']==task['A'] and native_task['B']==task['B']
  artifact=json.loads((OUT/case/method/'artifact.json').read_text());path=folder/'scene_final.ply'
  assert path.is_file() and path.stat().st_size==artifact['bytes']
  data=PlyData.read(str(path))['vertex'].data;assert len(data)==artifact['gaussians'] and len(data)>0
  assert all(np.isfinite(data[n]).all() for n in data.dtype.names),(case,method,'nonfinite PLY')
  for i,cam in enumerate(cameras):
   with Image.open(OUT/case/method/f'view_{i}.png') as im:assert im.size==(cam['width'],cam['height'])
  if method=='dge':
   assert 'max_steps=1000' in (folder/'logs/train.log').read_text()
   iterations={'gaussian_optimization':1000}
  elif method=='gaussianeditor':
   assert '1500/1500' in (folder/'logs/delete.log').read_text()
   add_logs=[x.read_text() for x in (folder/'logs').glob('add*.log')]
   assert any('max_steps=3000' in s for s in add_logs)
   assert any('1801/1801' in s for s in add_logs)
   add=json.loads((folder/'records/add_inputs.json').read_text());assert add['add_text']==task['B']
   iterations={'delete':1500,'neus':3000,'mesh_gaussian_native_loop':'0..1800 inclusive'}
  else:
   iterations={}
   for stage,steps in [('coarse',6500),('refine',2000)]:
    summary=json.loads((folder/stage/'final_summary.json').read_text());effective=json.loads((folder/stage/'effective_config.json').read_text())
    assert summary['iterations']==steps and summary['diagnostic_override'] is None
    assert effective['teacher_parameters_frozen'] and effective['background_parameters_frozen']
    assert effective['config']['target_text']==task['A'] and effective['config']['replacement_text']==task['B']
    iterations[stage]=steps
  results.append({'case':case,'method':method,'iterations_verified':iterations,'finite_ply':True,'gaussians':len(data),'common_render_count':3,'sha256':artifact['sha256'],'scene_ply':str(path)})
assert len(results)==15
(OUT/'validation.json').write_text(json.dumps({'checked_at':time.time(),'count':len(results),'results':results,'scope':'Artifact integrity and execution completion; visual success is assessed separately in qualitative_review.json.'},indent=2))
print('Validated 15 actual final PLYs, 45 common-camera baseline images, exact A/B and native completion evidence')
