"""Assert that the PLY used by GE/DGE/DC equals the checkpoint used by EditSplat."""
import json,os
from pathlib import Path
import numpy as np
from plyfile import PlyData
import torch
root=Path(os.environ['ROOT'])
ply=root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
checkpoint=root/'outputs/reconstruction/chkpnt30000.pth'
params,iteration=torch.load(checkpoint,map_location='cpu')
assert iteration==30000 and len(params)==12
vertices=PlyData.read(ply)['vertex'].data
n=len(vertices);assert params[1].shape[0]==n
fields={'xyz':(params[1].detach().numpy(),['x','y','z']),
 'f_dc':(params[2].detach().transpose(1,2).reshape(n,-1).numpy(),[f'f_dc_{i}' for i in range(3)]),
 'f_rest':(params[3].detach().transpose(1,2).reshape(n,-1).numpy(),[f'f_rest_{i}' for i in range(45)]),
 'scale':(params[4].detach().numpy(),[f'scale_{i}' for i in range(3)]),
 'rotation':(params[5].detach().numpy(),[f'rot_{i}' for i in range(4)]),
 'opacity':(params[6].detach().numpy(),['opacity'])}
errors={group:max(float(np.max(np.abs(array[:,i]-vertices[name]))) for i,name in enumerate(names)) for group,(array,names) in fields.items()}
assert all(error==0 for error in errors.values()),errors
report={'ply':str(ply),'checkpoint':str(checkpoint),'iteration':iteration,'gaussians':n,'active_sh_degree':params[0],'maximum_absolute_differences':errors,'status':'all_source_gaussian_parameters_exactly_equal'}
(root/'records/common_source_identity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
