"""Render every available result on identical, original calibrated cameras."""
import os,sys,json,argparse,math,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps,ImageFont
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison'
sys.path.insert(0,str(BASE/'vendor/gsle'));import common as c
p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--method');p.add_argument('--memory-fraction',type=float);a=p.parse_args()
if a.memory_fraction is not None:torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
spec=json.loads((BASE/'prompts.json').read_text());allcams={cam.name:cam for cam in c.source_cameras(938)}
OUT.mkdir(exist_ok=True)
def get_cameras(task):
 path=OUT/task['case']/'cameras.json';path.parent.mkdir(exist_ok=True)
 if path.exists():return [c.Camera.from_record(x) for x in json.loads(path.read_text())['cameras']]
 geom=json.loads(Path(task['instance_geometry']).read_text());center=np.array(geom['center'])
 half=np.array(geom['half_extents']);corners=c.bbox_corners(center-half*1.25,center+half*1.25)
 ranked=json.loads((Path(task['native_view_source'])/'selection.json').read_text())['ranking']
 main=allcams[task['reference_image']];selected=[main];candidates=[]
 for area,n in ranked:
  if area<100 or n not in allcams:continue
  cam=allcams[n];uv,z=cam.project(corners)
  if np.min(z)>0 and np.all(uv.min(0)>[cam.width*.03,cam.height*.03]) and np.all(uv.max(0)<[cam.width*.97,cam.height*.97]):candidates.append((area,cam))
 assert len(candidates)>=3, 'Not enough fully framed original cameras'
 def direction(cam):
  v=cam.center-center;return v/np.linalg.norm(v)
 maxarea=max(area for area,cam in candidates)
 while len(selected)<3:
  scores=[]
  for area,cam in candidates:
   if cam.name in [s.name for s in selected]:continue
   angles=[math.degrees(math.acos(np.clip(direction(cam)@direction(s),-1,1))) for s in selected]
   scores.append((min(min(angles),30)*math.sqrt(area/maxarea),cam))
  selected.append(max(scores,key=lambda v:v[0])[1])
 path.write_text(json.dumps({'policy':'Confirmed original reference plus two original registered cameras with visible old target and angular diversity; auxiliary cameras must fully contain the 1.25x old-target box with 3% frame margins. Input-only selection, identical camera and crop for every method.','cameras':[cam.record() for cam in selected]},indent=2))
 return selected
def ply_record(path):
 sha=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
 return {'path':str(path),'sha256':sha.hexdigest(),'bytes':path.stat().st_size}
for task in spec['cases']:
 if a.case and a.case!=task['case']:continue
 out=OUT/task['case'];cams=get_cameras(task)
 sources={'original':Path(spec['source_scene']),'gaussianeditor':BASE/'gaussianeditor'/task['case']/'scene_final.ply','dge':BASE/'dge'/task['case']/'scene_final.ply','gsle':BASE/'gsle'/task['case']/'scene_final.ply','ours':Path(task['ours_scene'])}
 for method,path in sources.items():
  if a.method and a.method!=method:continue
  if not path.is_file():continue
  dest=out/method;dest.mkdir(exist_ok=True)
  if (dest/'artifact.json').exists():continue
  print('render',task['case'],method,path,flush=True)
  data=c.GaussianData.load(path)
  with torch.no_grad():
   for i,cam in enumerate(cams):c.save_image(dest/f'view_{i}.png',c.render(cam,data,aux=False)['rgb'])
  points=len(data.xyz);del data;torch.cuda.empty_cache()
  record=ply_record(path);record['gaussians']=points
  (dest/'artifact.json').write_text(json.dumps(record,indent=2))
 # Preserve full-frame and identically cropped views, including missing-result labels.
 geom=json.loads(Path(task['instance_geometry']).read_text());center=np.array(geom['center']);half=np.array(geom['half_extents'])
 corners=c.bbox_corners(center-half,center+half)
 methods=list(sources);labels=['Original','GaussianEditor','DGE','GSLE (reproduction)','Ours']
 for crop in [False,True]:
  cellw=938 if not crop else 600;cellh=528 if not crop else 600
  board=Image.new('RGB',(5*cellw,3*(cellh+40)),(245,245,245));draw=ImageDraw.Draw(board)
  font=ImageFont.truetype('DejaVuSans.ttf',20 if not crop else 16)
  for i,cam in enumerate(cams):
   uv,z=cam.project(corners);lo=uv.min(0);hi=uv.max(0);mid=(lo+hi)/2;side=max(hi-lo)*2.0
   box=[int(max(0,mid[0]-side/2)),int(max(0,mid[1]-side/2)),int(min(cam.width,mid[0]+side/2)),int(min(cam.height,mid[1]+side/2))]
   for j,(method,label) in enumerate(zip(methods,labels)):
    x=j*cellw;y=i*(cellh+40);draw.text((x+9,y+10),f'{label} | {cam.name}',fill='black',font=font)
    path=out/method/f'view_{i}.png'
    if path.exists():
     im=Image.open(path).convert('RGB')
     if crop:im=im.crop(box)
     im=ImageOps.contain(im,(cellw,cellh));board.paste(im,(x+(cellw-im.width)//2,y+40+(cellh-im.height)//2))
    else:draw.text((x+30,y+100),'Result not yet available',fill='black')
  stem='comparison_crops' if crop else 'comparison_full'
  board.save(out/(stem+'.png'),dpi=(300,300))
  ImageOps.contain(board,(2500,2500)).save(out/(stem+'.jpg'),quality=95)
print('Comparison renders updated',flush=True)
