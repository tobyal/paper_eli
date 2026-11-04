"""Materialize official pinned Git requirements once for repeatable installs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import parse_qs

root=Path(os.environ['ROOT'])
source=root/'repos/GaussianEditor/requirements.lock.txt'
lines=source.read_text().splitlines()
entries=[]
for line in lines:
    if not line.startswith('git+'):
        continue
    base,_,fragment=line[4:].partition('#')
    url,revision=base.rsplit('@',1)
    url=url.rstrip('/')
    name=url.rsplit('/',1)[-1].removesuffix('.git')
    path=root/'repos/dependencies'/name
    subdir=parse_qs(fragment).get('subdirectory',[''])[0]
    entries.append((line,url,revision,path,subdir))

def checkout(entry):
    line,url,revision,path,subdir=entry
    path.mkdir(parents=True,exist_ok=True)
    def git(*args):
        return subprocess.check_output(['git','-C',str(path),*args],stderr=subprocess.STDOUT,text=True).strip()
    if not (path/'.git').exists():
        git('init')
        git('remote','add','origin',url)
    if not (path/'.revision_ready').exists():
        git('fetch','--depth','1','origin',revision)
        git('checkout','--detach','FETCH_HEAD')
        git('submodule','update','--init','--recursive','--depth','1')
        (path/'.revision_ready').write_text(git('rev-parse','HEAD'))
    commit=git('rev-parse','HEAD')
    print(f'{path.name}: {commit}',flush=True)
    return line,str(path/subdir),{'url':url,'requested_revision':revision,'commit':commit,'path':str(path)}

with ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(checkout,entries))
replacements={line:path for line,path,_ in results}
(root/'records/ge_requirements_local.txt').write_text('\n'.join(replacements.get(line,line) for line in lines)+'\n')
(root/'records/ge_git_dependencies.json').write_text(json.dumps([record for _,_,record in results],indent=2))
