"""Replay official semantic tracing on saved SAM masks; retain exact weights."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
root=Path(os.environ['ROOT'])
sys.path.insert(0,str(root/'repos/GaussianEditor'))
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.camera_scene import CamScene

model=GaussianModel(sh_degree=0,anchor_weight_init_g0=.01,anchor_weight_init=.01,anchor_weight_multiplier=1.3)
model.load_ply(str(root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'))
scene=CamScene(str(root/'data/edit_train'),h=512,w=512)
cache=next((root/'repos/GaussianEditor/edit_cache/industrial_red_gaussianeditor').rglob('gs_mask.pt')).parent
weights=torch.zeros_like(model._opacity)
counts=torch.zeros_like(model._opacity,dtype=torch.int32)
with torch.no_grad():
    for path in sorted(cache.glob('*.png')):
        if not path.stem.isdigit(): continue
        mask=np.array(Image.open(path))[:,:,0]/255.
        mask=torch.tensor(mask,device='cuda',dtype=torch.float32)[None]
        model.apply_weights(scene.cameras[int(path.stem)],weights,counts,mask)
weights=weights/(counts+1e-7)
torch.save({'weights':weights.cpu(),'counts':counts.cpu()},cache/'tracing_weights.pt')
geometry=json.loads((root/'records/target_geometry.json').read_text())
center=torch.tensor(geometry['center'],device='cuda',dtype=torch.float32)
basis=torch.tensor(geometry['basis'],device='cuda',dtype=torch.float32)
half=torch.tensor(geometry['half_extents'],device='cuda',dtype=torch.float32)
roi=(((model.get_xyz-center)@basis.T).abs()<=half).all(dim=1)
report={'target_gaussians':int(roi.sum()),'target_weight_quantiles':torch.quantile(weights[roi,0],torch.tensor([0.,.1,.25,.5,.75,.9,1.],device='cuda')).tolist(),'thresholds':[]}
for threshold in [.1,.2,.3,.4,.5,.6,.8]:
    selected=weights[:,0]>threshold
    torch.save(selected.cpu(),cache/f'gs_mask_threshold_{threshold:.1f}.pt')
    report['thresholds'].append({'threshold':threshold,'selected':int(selected.sum()),'inside_target_box':int((selected&roi).sum())})
(root/'records/gaussianeditor_tracing_diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
