"""Reproduce DGE camera-ordering numerical-domain issue on actual input poses."""
import importlib.util,json,os,random
from pathlib import Path
import numpy as np
root=Path(os.environ['ROOT'])
spec=importlib.util.spec_from_file_location('colmap_io',root/'repos/DGE/gaussiansplatting/scene/colmap_loader.py')
io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)
ims=sorted(io.read_extrinsics_text(str(root/'data/edit_train/sparse/0/images.txt')).values(),key=lambda r:r.name)
random.seed(0)
ims=[ims[i] for i in random.sample(range(len(ims)),20)]
Rs=[io.qvec2rotmat(im.qvec).T for im in ims]
centers=[-R@im.tvec for R,im in zip(Rs,ims)]
left=int(np.argmin(np.array(centers)[:,0]));forward=Rs[left][:,2]
dots=[float(forward@R[:,2]) for R in Rs]
out={'left_reference':ims[left].name,'dot_products':dots,'out_of_domain':[{'frame':im.name,'dot':dot} for im,dot in zip(ims,dots) if abs(dot)>1], 'max_rotation_orthogonality_error':max(float(np.abs(R@R.T-np.eye(3)).max()) for R in Rs)}
(root/'records/dge_angle_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
