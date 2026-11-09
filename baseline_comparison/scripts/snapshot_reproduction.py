"""Capture exact executed source and cumulative upstream patches, without weights."""
import hashlib,io,json,subprocess,tarfile,zipfile,difflib
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
RECORDS=BASE/'records';EXPORT=BASE/'comparison';EXPORT.mkdir(exist_ok=True)
suffixes={'.py','.sh','.yaml','.yml','.json','.toml','.ini','.cpp','.cu','.h','.hpp','.cuh','.c','.md','.txt','.cfg','.in'}
files={};manifest={};patches={}
def add(path,label):
 if path.is_file() and not path.is_symlink():
  b=path.read_bytes();files[label]=b
  manifest[label]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
for method,folder in [('GaussianEditor',BASE/'vendor/GaussianEditor'),('DGE',ROOT/'repos/DGE')]:
 repo=ROOT/'repos'/method
 commit=(RECORDS/(method+'_commit.txt')).read_text().strip()
 raw=subprocess.check_output(['git','archive',commit],cwd=repo)
 patch=[]
 with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
  names=[]
  for member in archive:
   if not member.isfile():continue
   name=member.name;path=folder/name
   if Path(name).suffix.lower() not in suffixes and Path(name).name not in ['LICENSE','Makefile','Dockerfile']:continue
   names.append(name);old=archive.extractfile(member).read()
   new=path.read_bytes() if path.is_file() else b''
   add(path,'source/'+method+'/'+name)
   if old!=new:
    try:patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    except UnicodeDecodeError:patch.append('Binary file changed: '+name+'\n')
 for name in ['instance_selection.py']:
  path=folder/name;add(path,'source/'+method+'/'+name)
  if path.is_file():patch.extend(difflib.unified_diff([],path.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
 (RECORDS/(method+'_final_cumulative.patch')).write_text(''.join(patch))
for path in sorted((BASE/'vendor/gsle').rglob('*')):
 if '__pycache__' not in path.parts and path.suffix in suffixes:add(path,'source/gsle/'+str(path.relative_to(BASE/'vendor/gsle')))
for path in sorted((BASE/'scripts').glob('*.py')):add(path,'baseline_comparison/scripts/'+path.name)
for name in ['prompts.json','README.md']:add(BASE/name,'baseline_comparison/'+name)
for path in sorted(RECORDS.iterdir()):
 if path.is_file() and path.suffix in suffixes and path.name!='executed_source_manifest.json':add(path,'baseline_comparison/records/'+path.name)
for case in ['case00','case02','case04','case07','case10']:
 for method in ['gaussianeditor','dge','gsle']:
  folder=BASE/method/case
  for name in ['task.json','config.json']:
   add(folder/name,f'baseline_comparison/{method}/{case}/{name}')
  for path in sorted((folder/'records').glob('*.json')):
   if path.name not in ['coarse_progress.json','refine_progress.json']:add(path,f'baseline_comparison/{method}/{case}/records/{path.name}')
 add(EXPORT/case/'cameras.json',f'baseline_comparison/comparison/{case}/cameras.json')
source_record={k:v for k,v in manifest.items() if k.startswith('source/') or '/scripts/' in k}
(RECORDS/'executed_source_manifest.json').write_text(json.dumps(source_record,indent=2))
with zipfile.ZipFile(EXPORT/'reproduction_code_and_records.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for label,b in files.items():z.writestr(label,b)
 z.writestr('manifest_sha256.json',json.dumps(manifest,indent=2))
print('Saved',len(source_record),'source files;', (EXPORT/'reproduction_code_and_records.zip').stat().st_size,'archive bytes')
