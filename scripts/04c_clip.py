"""Use OpenAI's published CLIP checkpoint and checksum for common evaluation."""
import ast
import json
import os
from pathlib import Path
import subprocess

root=Path(os.environ['ROOT'])
source=root/'repos/dependencies/CLIP/clip/clip.py'
tree=ast.parse(source.read_text())
models=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_MODELS' for target in node.targets))
url=models['ViT-L/14']
digest=url.split('/')[-2]
destination=root/'models/clip'
destination.mkdir(parents=True,exist_ok=True)
subprocess.run(['aria2c','--continue=true','--auto-file-renaming=false','--max-connection-per-server=8','--split=8','--min-split-size=32M','--file-allocation=none','--summary-interval=30','--console-log-level=warn',f'--all-proxy={os.environ["https_proxy"]}',f'--checksum=sha-256={digest}',f'--dir={destination}',url],check=True)
(root/'records/clip_model.json').write_text(json.dumps({'model':'ViT-L/14','url':url,'sha256':digest,'path':str(destination/url.split('/')[-1])},indent=2))
