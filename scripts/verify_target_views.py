"""Render the unchanged source model in calibrated target cameras for alignment QA."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
root=Path(os.environ['ROOT'])
sys.path.insert(0,str(root/'repos/gaussian-splatting'))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import getWorld2View2,getProjectionMatrix,focal2fov
manifest=json.loads((root/'records/target_view_manifest.json').read_text())
model=GaussianModel(3)
model.load_ply(str(root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'))
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
background=torch.zeros(3,device='cuda')
out=root/'outputs/target_view_check'
out.mkdir(exist_ok=True)
rows=[]
with torch.no_grad():
 for r in manifest['views']:
  if r['frame'] not in ['frame_000058.jpg','frame_000151.jpg','frame_000160.jpg','frame_000275.jpg']: continue
  view=torch.tensor(getWorld2View2(qvec2rotmat(np.array(r['qvec'])).T,np.array(r['tvec'])),device='cuda').T
  fov=focal2fov(r['focal'],512)
  projection=getProjectionMatrix(znear=.01,zfar=100,fovX=fov,fovY=fov).cuda().T
  camera=MiniCam(512,512,fov,fov,.01,100,view,view@projection)
  rgb=render(camera,model,pipe,background)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
  real=np.array(Image.open(root/f'data/edit_{r["split"]}_target/images'/r['frame']).convert('RGB'))/255.
  psnr=float(-10*np.log10(np.mean((real-rgb)**2)))
  Image.fromarray((rgb*255).round().astype(np.uint8)).save(out/Path(r['frame']).with_suffix('.png'))
  Image.fromarray((np.concatenate([real,rgb],axis=1)*255).round().astype(np.uint8)).save(out/(Path(r['frame']).stem+'_pair.png'))
  rows.append({'frame':r['frame'],'psnr_real_vs_source_render':psnr})
(out/'alignment.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
