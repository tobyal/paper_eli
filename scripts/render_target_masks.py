"""Render one fixed, independently annotated physical target for evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

root=Path(os.environ['ROOT'])
sys.path.insert(0,str(root/'repos/gaussian-splatting'))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat
from utils.graphics_utils import getWorld2View2,getProjectionMatrix,focal2fov
parser=argparse.ArgumentParser()
parser.add_argument('--iteration',type=int,default=30000)
parser.add_argument('--name',default='evaluation_masks')
args=parser.parse_args()
destination=root/'outputs'/args.name
(destination/'wide').mkdir(parents=True,exist_ok=True)
(destination/'overlays').mkdir(parents=True,exist_ok=True)
geometry=json.loads((root/'records/target_geometry.json').read_text())
model=GaussianModel(3)
model.load_ply(str(root/f'outputs/reconstruction/point_cloud/iteration_{args.iteration}/point_cloud.ply'))
center=torch.tensor(geometry['center'],device='cuda',dtype=torch.float32)
basis=torch.tensor(geometry['basis'],device='cuda',dtype=torch.float32)
half=torch.tensor(geometry['half_extents'],device='cuda',dtype=torch.float32)
selected=(((model.get_xyz-center)@basis.T).abs()<=half).all(dim=1)
colors=selected.float()[:,None].repeat(1,3)
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
background=torch.zeros(3,device='cuda')
cameras=read_intrinsics_binary(str(root/'data/industrial/sparse/0/cameras.bin'))
images=read_extrinsics_binary(str(root/'data/industrial/sparse/0/images.bin'))
records=[]
with torch.no_grad():
  for index,im in enumerate(sorted(images.values(),key=lambda item:item.name)):
    intr=cameras[im.camera_id]
    w,h=960,2*round(intr.height*960/intr.width/2)
    fx,fy=intr.params[0]*w/intr.width,intr.params[1]*h/intr.height
    fovx,fovy=focal2fov(fx,w),focal2fov(fy,h)
    R=qvec2rotmat(im.qvec)
    view=torch.tensor(getWorld2View2(R.T,im.tvec),device='cuda').transpose(0,1)
    projection=getProjectionMatrix(znear=.01,zfar=100,fovX=fovx,fovY=fovy).cuda().transpose(0,1)
    camera=MiniCam(w,h,fovy,fovx,.01,100,view,view@projection)
    weight=render(camera,model,pipe,background,override_color=colors)['render'][0].cpu().numpy()
    mask=weight>.25
    name=Path(im.name).stem
    Image.fromarray((mask*255).astype(np.uint8)).save(destination/'wide'/(name+'.png'))
    records.append({'frame':im.name,'pixels':int(mask.sum())})
    if mask.sum()>100 and (index%10==0 or index==150):
        rgb=render(camera,model,pipe,background)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
        rgb[mask]=rgb[mask]*.5+np.array([.5,0,0])
        Image.fromarray((rgb*255).astype(np.uint8)).save(destination/'overlays'/(name+'.jpg'))
  report={'source_iteration':args.iteration,'selected_gaussians':int(selected.sum()),'total_gaussians':len(selected),'mask_threshold':.25,'cameras':records,'status':'requires_visual_review'}
  (destination/'mask_manifest.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({key:value for key,value in report.items() if key!='cameras'}),flush=True)
  print('Visible views:',sum(row['pixels']>100 for row in records),flush=True)
