"""Lift a visually reviewed reference SAM mask into the shared 3D coordinates."""
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

model=GaussianModel(3)
model.load_ply(str(root/'outputs/reconstruction/point_cloud/iteration_7000/point_cloud.ply'))
extrinsics=read_extrinsics_binary(str(root/'data/industrial/sparse/0/images.bin'))
intrinsics=read_intrinsics_binary(str(root/'data/industrial/sparse/0/cameras.bin'))
im=next(im for im in extrinsics.values() if im.name=='frame_000151.jpg')
intr=intrinsics[im.camera_id]
fx,fy,cx,cy=intr.params
side=2*min(cx,cy,intr.width-cx,intr.height-cy)
fx,fy=fx*512/side,fy*512/side
R=qvec2rotmat(im.qvec)
view=torch.tensor(getWorld2View2(R.T,im.tvec),device='cuda').transpose(0,1)
projection=getProjectionMatrix(znear=.01,zfar=100,fovX=focal2fov(fx,512),fovY=focal2fov(fy,512)).cuda().transpose(0,1)
camera=MiniCam(512,512,focal2fov(fy,512),focal2fov(fx,512),.01,100,view,view@projection)
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
background=torch.zeros(3,device='cuda')
with torch.no_grad():
    z=(model.get_xyz@torch.tensor(R,device='cuda',dtype=torch.float32).T+torch.tensor(im.tvec,device='cuda',dtype=torch.float32))[:,2]
    packed=torch.stack([z,torch.ones_like(z),torch.zeros_like(z)],dim=1)
    output=render(camera,model,pipe,background,override_color=packed)['render']
    depth=(output[0]/output[1].clamp_min(1e-6)).cpu().numpy()
    alpha=output[1].cpu().numpy()
mask=np.array(Image.open(root/'outputs/reference_sam/mask_0.png'))>127
valid=mask&(alpha>.95)&(depth>0)
y,x=np.where(valid)
z=depth[valid]
points=np.stack([(x-256)/fx*z,(y-256)/fy*z,z],axis=1)
points=(points-im.tvec)@R
center=np.median(points,axis=0)
_,_,basis=np.linalg.svd(points-center,full_matrices=False)
local=(points-center)@basis.T
low,high=np.percentile(local,[2,98],axis=0)
center=center+((low+high)/2)@basis
half=(high-low)/2
half=half+np.maximum(half*.12,half.max()*.025)
record={'reference_frame':im.name,'reference_mask':'outputs/reference_sam/mask_0.png',
 'review':'SAM candidate0 visually isolates white lower housing; candidates1/2 additionally include vertical posts.',
 'source_geometry_iteration':7000,'center':center.tolist(),'basis':basis.tolist(),'half_extents':half.tolist(),
 'point_count':len(points),'method':'Alpha-normalized source Gaussian depth; robust PCA bounds using 2nd/98th percentiles plus 12% margin. This annotation disambiguates repeated devices and defines common evaluation ROI.'}
(root/'records/target_geometry.json').write_text(json.dumps(record,indent=2))
np.savez_compressed(root/'records/target_reference_points.npz',points=points)
print(json.dumps(record,indent=2))
