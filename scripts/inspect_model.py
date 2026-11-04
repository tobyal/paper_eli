"""Check saved Gaussian parameters and record the exact artifact hash."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData

parser=argparse.ArgumentParser()
parser.add_argument('ply',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
vertices=PlyData.read(args.ply)['vertex'].data
assert len(vertices)>0
nonfinite={name:int((~np.isfinite(vertices[name])).sum()) for name in vertices.dtype.names if np.issubdtype(vertices.dtype[name],np.floating)}
assert sum(nonfinite.values())==0, nonfinite
xyz=np.column_stack([vertices[name] for name in ['x','y','z']])
rotation=np.column_stack([vertices[f'rot_{i}'] for i in range(4)])
assert np.all(np.linalg.norm(rotation,axis=1)>1e-12), 'Invalid zero quaternion'
digest=hashlib.sha256()
with args.ply.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
report={'path':str(args.ply.resolve()),'bytes':args.ply.stat().st_size,'sha256':digest.hexdigest(),'gaussians':len(vertices),'all_parameters_finite':True,'all_quaternions_nonzero':True,'xyz_min':xyz.min(axis=0).tolist(),'xyz_max':xyz.max(axis=0).tolist(),'log_scale_range':[float(min(vertices[f'scale_{i}'].min() for i in range(3))),float(max(vertices[f'scale_{i}'].max() for i in range(3)))],'opacity_logit_range':[float(vertices['opacity'].min()),float(vertices['opacity'].max())]}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
