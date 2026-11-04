"""Analyze actual saved 2D targets; these are diagnostics, not final3D scores."""
import json,os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
root=Path(os.environ['ROOT']);out=root/'outputs/editsplat'
ranking=json.loads((out/'imagereward_ranking.json').read_text())
transforms=json.loads((root/'records/edit_image_transform.json').read_text())
# This scene has one calibrated PINHOLE camera; read actual source dimensions.
cameras=json.loads((root/'outputs/reconstruction/common_render/cameras.json').read_text())
assert all(c['width']==960 and c['height']==540 for c in cameras)
affine=next(iter(transforms['affine_by_camera'].values()))
sx,sy=960/1875,540/1056
mask_transform=(affine[0]*sx,0,affine[2]*sx,0,affine[4]*sy,affine[5]*sy)
def red(array):
 hsv=cv2.cvtColor(array.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
 return ((hsv[:,:,0]<24)|(hsv[:,:,0]>336))&(hsv[:,:,1]>.3)&(hsv[:,:,2]>.15)
result={'role':'Diagnostic2D diffusion targets; final3D quality is scored separately on held-out renders','mask':'Same approximate3D target mask resampled into calibrated512 input crop','stages':{}}
for stage in ['initial_2d','mfg_2d']:
 rows=[]
 for i,name in enumerate(ranking['camera_names']):
  file=out/stage/f'{i:06d}.png'
  if not file.exists():continue
  image=np.array(Image.open(file).convert('RGB'))
  mask=np.array(Image.open(root/'outputs/evaluation_masks/wide'/(name+'.png')).transform((512,512),Image.Transform.AFFINE,mask_transform,Image.Resampling.NEAREST))>127
  redmask=red(image)
  row={'index':i,'camera':name,'rank':ranking['ranking'][i],'selected':ranking['selected'][i],'target_pixels':int(mask.sum()),'global_red_fraction':float(redmask.mean()),'outside_target_red_fraction':float(redmask[~mask].mean())}
  if mask.sum()>=100:row['target_red_fraction']=float(redmask[mask].mean())
  rows.append(row)
 summary={}
 for label,subset in [('all',rows),('selected',[r for r in rows if r['selected']])]:
  valid=[r for r in subset if 'target_red_fraction' in r]
  summary[label]={'views':len(subset),'target_visible_views':len(valid),'mean_global_red_fraction':float(np.mean([r['global_red_fraction'] for r in subset])) if subset else None,'mean_target_red_fraction':float(np.mean([r['target_red_fraction'] for r in valid])) if valid else None,'mean_outside_target_red_fraction':float(np.mean([r['outside_target_red_fraction'] for r in subset])) if subset else None}
 result['stages'][stage]={'summary':summary,'views':rows}
(root/'records/editsplat_2d_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v['summary'] for k,v in result['stages'].items()},indent=2))
