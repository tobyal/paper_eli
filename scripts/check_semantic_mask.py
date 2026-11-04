"""Measure the official traced Gaussian mask against the reviewed physical ROI."""
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
method=sys.argv[1]
repos={'gaussianeditor':'GaussianEditor','dge':'DGE','dreamcatalyst':'DreamCatalyst/threestudio'}
cache=root/'repos'/repos[method]/f'edit_cache/industrial_red_{method}'
path=Path(sys.argv[2]) if len(sys.argv)>2 else next(cache.rglob('gs_mask.pt'))
selected=torch.load(path,map_location='cuda')
model=GaussianModel(3)
model.load_ply(str(root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'))
assert len(selected)==len(model.get_xyz)
colors=selected.float()[:,None].repeat(1,3)
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
background=torch.zeros(3,device='cuda')
cameras=read_intrinsics_binary(str(root/'data/industrial/sparse/0/cameras.bin'))
images=read_extrinsics_binary(str(root/'data/industrial/sparse/0/images.bin'))
destination=root/'outputs'/method/('segmentation_review_'+sys.argv[3] if len(sys.argv)>3 else 'segmentation_review')
destination.mkdir(parents=True,exist_ok=True)
rows=[]
with torch.no_grad():
  for im in sorted(images.values(),key=lambda item:item.name):
    if im.name not in [f'frame_{i:06d}.jpg' for i in [61,91,151,161,281]]: continue
    intr=cameras[im.camera_id]
    w,h=960,540
    fx,fy=intr.params[0]*w/intr.width,intr.params[1]*h/intr.height
    fovx,fovy=focal2fov(fx,w),focal2fov(fy,h)
    view=torch.tensor(getWorld2View2(qvec2rotmat(im.qvec).T,im.tvec),device='cuda').transpose(0,1)
    projection=getProjectionMatrix(znear=.01,zfar=100,fovX=fovx,fovY=fovy).cuda().transpose(0,1)
    camera=MiniCam(w,h,fovy,fovx,.01,100,view,view@projection)
    weight=render(camera,model,pipe,background,override_color=colors)['render'][0].cpu().numpy()
    mask=weight>.25
    name=Path(im.name).with_suffix('.png')
    target=np.array(Image.open(root/'outputs/evaluation_masks/wide'/name))>127
    rows.append({'frame':im.name,'selected_pixels':int(mask.sum()),'target_pixels':int(target.sum()),'coverage':float((mask&target).sum()/max(1,target.sum())),'precision':float((mask&target).sum()/max(1,mask.sum()))})
    rgb=np.array(Image.open(root/'outputs/reconstruction/common_render/wide'/name)).astype(np.float32)
    rgb[mask]=rgb[mask]*.5+np.array([127.5,0,0])
    Image.fromarray(rgb.astype(np.uint8)).save(destination/(Path(im.name).stem+'.jpg'))
report={'source_mask':str(path),'selected_gaussians':int(selected.sum()),'views':rows}
(destination/'review.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
