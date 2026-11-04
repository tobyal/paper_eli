"""Pin the exact upstream packages used in the verified EditSplat environment."""
import os,subprocess
from pathlib import Path
root=Path(os.environ['ROOT']);repo=root/'repos/dependencies/lang-sam-editsplat'
commit='adc57a72c317e777621e613da30532939be7c0d6'
if not (repo/'.git').exists():
 subprocess.run(['git','clone','https://github.com/paulguerrero/lang-sam.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',commit],check=True)
path=repo/'pyproject.toml';text=path.read_text()
for package,url,pin in [
 ('segment-anything','https://github.com/facebookresearch/segment-anything.git','dca509fe793f601edb92606367a655c15ac00fdf'),
 ('groundingdino','https://github.com/IDEA-Research/GroundingDINO.git','856dde20aee659246248e20734ef9ba5214f5e44')]:
 old=package+' = {git = "'+url+'"}'
 new=package+' = {git = "'+url+'", rev = "'+pin+'"}'
 assert old in text or new in text
 text=text.replace(old,new)
path.write_text(text)
(root/'records/editsplat_langsam_dependency_pins.patch').write_text(subprocess.check_output(['git','diff','--','pyproject.toml'],cwd=repo,text=True))
print('Pinned LangSAM, GroundingDINO and SAM to installed revisions')
