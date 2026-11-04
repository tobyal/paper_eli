"""Render every method using one unchanged original 3DGS rasterizer/camera path."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

root=Path(os.environ['ROOT'])
sys.path.insert(0,str(root/'repos/gaussian-splatting'))
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat
from utils.graphics_utils import getWorld2View2,getProjectionMatrix,focal2fov

parser=argparse.ArgumentParser()
parser.add_argument('--ply',type=Path,required=True)
parser.add_argument('--name',required=True)
parser.add_argument('--width',type=int,default=960)
parser.add_argument('--depth',action='store_true')
parser.add_argument('--frames',help='Comma-separated input frame numbers, for reconstruction previews only')
args=parser.parse_args()
destination=root/'outputs'/args.name/'common_render'
wide=destination/'wide'
square=destination/'test_512'
wide.mkdir(parents=True,exist_ok=True)
square.mkdir(parents=True,exist_ok=True)
vertex=PlyData.read(args.ply)['vertex']
rest=sum(name.startswith('f_rest_') for name in vertex.data.dtype.names)
degree=int(round(math.sqrt(rest//3+1)-1))
gaussians=GaussianModel(degree)
gaussians.load_ply(str(args.ply))
del vertex
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
background=torch.zeros(3,device='cuda')
cameras=read_intrinsics_binary(str(root/'data/industrial/sparse/0/cameras.bin'))
extrinsics=read_extrinsics_binary(str(root/'data/industrial/sparse/0/images.bin'))
ordered=sorted(extrinsics.values(),key=lambda im:im.name)
if args.frames:
    wanted={f'frame_{int(value):06d}.jpg' for value in args.frames.split(',')}
    ordered=[im for im in ordered if im.name in wanted]
test_names=set(json.loads((root/'records/camera_split.json').read_text())['test'])
records=[]
def camera(im,intr,w,h,cropped=False):
    fx,fy,cx,cy=intr.params
    if cropped:
        side=2*min(cx,cy,intr.width-cx,intr.height-cy)
        fx,fy=fx*w/side,fy*h/side
    else:
        fx,fy=fx*w/intr.width,fy*h/intr.height
    fovx,fovy=focal2fov(fx,w),focal2fov(fy,h)
    R=qvec2rotmat(im.qvec)
    view=torch.tensor(getWorld2View2(R.T,im.tvec),device='cuda').transpose(0,1)
    projection=getProjectionMatrix(znear=0.01,zfar=100.0,fovX=fovx,fovY=fovy).to('cuda').transpose(0,1)
    cam=MiniCam(w,h,fovy,fovx,0.01,100.0,view,view@projection)
    return cam,{'name':im.name,'R':R.tolist(),'t':im.tvec.tolist(),'fx':float(fx),'fy':float(fy),'cx':w/2,'cy':h/2,'width':w,'height':h}

with torch.no_grad():
    for index,im in enumerate(ordered):
        intr=cameras[im.camera_id]
        assert intr.model=='PINHOLE'
        w=args.width
        h=2*round(intr.height*w/intr.width/2)
        cam,record=camera(im,intr,w,h)
        rgb=render(cam,gaussians,pipe,background)['render'].clamp(0,1)
        Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(wide/(Path(im.name).stem+'.png'))
        record['split']='test' if im.name in test_names else 'train'
        records.append(record)
        if args.depth:
            R=torch.tensor(record['R'],device='cuda',dtype=torch.float32)
            t=torch.tensor(record['t'],device='cuda',dtype=torch.float32)
            z=(gaussians.get_xyz@R.T+t)[:,2]
            packed=torch.stack([z,torch.ones_like(z),torch.zeros_like(z)],dim=-1)
            values=render(cam,gaussians,pipe,background,override_color=packed)['render']
            depth=(values[0]/values[1].clamp_min(1e-6)).cpu().numpy()
            np.savez_compressed(wide/(Path(im.name).stem+'.npz'),depth=depth.astype(np.float16),alpha=values[1].cpu().numpy().astype(np.float16))
        if im.name in test_names:
            cam512,_=camera(im,intr,512,512,True)
            rgb512=render(cam512,gaussians,pipe,background)['render'].clamp(0,1)
            Image.fromarray((rgb512.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(square/(Path(im.name).stem+'.png'))
        if index%30==0:
            print(f'{args.name}: {index+1}/{len(ordered)} cameras',flush=True)

(destination/'cameras.json').write_text(json.dumps(records,indent=2))
(destination/'model.json').write_text(json.dumps({'source':str(args.ply),'gaussians':len(gaussians.get_xyz),'sh_degree':degree,'renderer':'original graphdeco 3DGS','trajectory':'registered source-video cameras in temporal order','width':args.width},indent=2))
if not args.frames:
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-framerate','12','-pattern_type','glob','-i',str(wide/'*.png'),'-c:v','libx264','-threads','8','-crf','18','-pix_fmt','yuv420p',str(destination/'multiview.mp4')],check=True)
print(destination,flush=True)
