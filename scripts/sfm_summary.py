import importlib.util
import json
import os
from pathlib import Path
import numpy as np

root=Path(os.environ['ROOT'])
spec=importlib.util.spec_from_file_location('io',root/'repos/GaussianEditor/gaussiansplatting/scene/colmap_loader.py')
io=importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
sparse=root/'data/industrial/sparse/0'
images=io.read_extrinsics_binary(str(sparse/'images.bin'))
cameras=io.read_intrinsics_binary(str(sparse/'cameras.bin'))
xyz,rgb,error=io.read_points3D_binary(str(sparse/'points3D.bin'))
ordered=sorted(images.values(),key=lambda im:im.name)
centers=np.stack([-io.qvec2rotmat(im.qvec).T@im.tvec for im in ordered])
names=[im.name for im in ordered]
all_names=[p.name for p in sorted((root/'data/industrial/input').glob('*.jpg'))]
report={'input_frames':len(all_names),'registered':len(names),'registration_fraction':len(names)/len(all_names),'unregistered':[name for name in all_names if name not in set(names)],'points':len(xyz),'point_reprojection_error_mean':float(np.mean(error)),'point_reprojection_error_median':float(np.median(error)),'cameras':[{**cam._asdict(),'params':cam.params.tolist()} for cam in cameras.values()]}
(root/'records/sfm_summary.json').write_text(json.dumps(report,indent=2))
sample=np.random.default_rng(0).choice(len(xyz),min(18000,len(xyz)),replace=False)
np.savez_compressed(root/'records/sfm_preview.npz',xyz=xyz[sample],rgb=rgb[sample],centers=centers,names=np.array(names))
print(json.dumps(report,indent=2))
