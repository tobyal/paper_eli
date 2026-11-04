"""Pin upstream dependencies that were floating in the 2023 LangSAM release."""
import os
from pathlib import Path
import subprocess
import json

root=Path(os.environ['ROOT'])
gd=root/'repos/dependencies/GroundingDINO'
commit='16e6b4bfcf6ed3a6f0ef2bc5752d716657a6b9fe'
if not (gd/'.git').exists():
    subprocess.run(['git','clone','--filter=blob:none','https://github.com/IDEA-Research/GroundingDINO.git',str(gd)],check=True)
subprocess.run(['git','-C',str(gd),'checkout','--detach',commit],check=True)
sam=root/'repos/dependencies/segment-anything'
if not (sam/'.git').exists():
    subprocess.run(['git','clone','https://github.com/facebookresearch/segment-anything.git',str(sam)],check=True)
sam_commit='6fdee8f2727f4506cfbbe553e23b895e27956588'
subprocess.run(['git','-C',str(sam),'checkout','--detach',sam_commit],check=True)
meta=root/'repos/dependencies/lang-segment-anything/pyproject.toml'
text=meta.read_text()
lines=[]
for line in text.splitlines():
    if line.startswith('groundingdino ='): line='groundingdino = {path = "'+str(gd)+'"}'
    if line.startswith('segment-anything ='): line='segment-anything = {path = "'+str(sam)+'"}'
    lines.append(line)
meta.write_text('\n'.join(lines)+'\n')
(root/'records/transitive_dependency_pins.json').write_text(json.dumps({'reason':'Floating GroundingDINO main now requires supervision>=0.22, incompatible with LangSAM Pillow 9.3.0; use official dependency versions contemporaneous with GE locked release. Only packaging references changed.','groundingdino':commit,'segment_anything':sam_commit},indent=2))
