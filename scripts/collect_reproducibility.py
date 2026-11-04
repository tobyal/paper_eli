"""Collect exact source changes and environment metadata after actual runs."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(os.environ['ROOT'])
def command(argv,cwd=None):
 result=subprocess.run(argv,cwd=cwd,capture_output=True,text=True)
 return {'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
repos=['GaussianEditor','DGE','DreamCatalyst','cvpr25_EditSplat','gaussian-splatting']
out=root/'records/reproducibility';out.mkdir(exist_ok=True)
manifest={'repositories':{}}
for name in repos:
 repo=root/'repos'/name
 entry={'head':command(['git','rev-parse','HEAD'],repo)['stdout'].strip(),
 'status':command(['git','status','--short'],repo)['stdout'],
 'submodules':command(['git','submodule','status','--recursive'],repo)['stdout']}
 (out/(name+'.patch')).write_text(command(['git','diff','--binary'],repo)['stdout'])
 for file in ['instance_selection.py','threestudio/instance_selection.py']:
  path=repo/file
  if path.exists():
   dest=out/'added_source'/name/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
 manifest['repositories'][name]=entry
manifest['platform']=command(['uname','-a'])
manifest['gpu']=command(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'])
manifest['cuda']=command(['/usr/local/cuda-11.8/bin/nvcc','--version'])
manifest['scripts_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'scripts').glob('*') if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Saved exact source patches, added selector files, hardware and script checksums')
