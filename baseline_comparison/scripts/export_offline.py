"""Portable inspection package; large scene PLY files stay on the SSH server."""
import re,json,zipfile,html
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';OUT=BASE/'comparison'
page=(OUT/'index.html').read_text()
def server_ply(match):
 path=(OUT/html.unescape(match.group(1))).resolve()
 return '<code title="Download with scp from ubuntuA6000">Server PLY: '+html.escape(str(path))+'</code>'
page=re.sub(r'<a href="([^"]+\.ply)">[^<]+</a>',server_ply,page)
page=page.replace('<h1>','<p>This offline copy contains figures, prompts and records. Final PLY files remain on ubuntuA6000; their full server paths are shown below.</p><h1>',1)
files=[]
for n in ['prompts.json','README.md']:files.append(BASE/n)
for path in OUT.iterdir():
 if path.is_file() and path.suffix in ['.json','.csv','.md','.png','.jpg']:files.append(path)
for case in ['case00','case02','case04','case07','case10']:
 folder=OUT/case
 for path in folder.rglob('*'):
  if path.is_file() and path.suffix in ['.png','.jpg','.json']:files.append(path)
 for method in ['gaussianeditor','dge','gsle']:
  folder=BASE/method/case
  for n in ['task.json','config.json']:files.append(folder/n)
  for name in ['records','logs']:
   files.extend(p for p in (folder/name).glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt','.yaml'])
files.append(OUT/'reproduction_code_and_records.zip')
with zipfile.ZipFile(OUT/'offline_review.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.writestr('comparison/index.html',page)
 for path in dict.fromkeys(files):
  if path.is_file():z.write(path,str(path.relative_to(BASE)))
print('Offline review bytes:',(OUT/'offline_review.zip').stat().st_size)
