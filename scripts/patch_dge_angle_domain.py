"""Numerical-domain fix only: clip cosine to[-1,1] before inverse cosine."""
import os,subprocess
from pathlib import Path
root=Path(os.environ['ROOT']);repo=root/'repos/DGE'
path=repo/'threestudio/systems/DGE.py'
text=path.read_text()
old='np.arccos(np.dot(most_left_vecotr, cam.R[:, 2]))'
new='np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), -1.0, 1.0))'
if old in text:
 assert text.count(old)==2
 text=text.replace(old,new)
 path.write_text(text)
assert text.count(new)==2
patch=subprocess.check_output(['git','diff','--','threestudio/systems/DGE.py'],cwd=repo,text=True)
(root/'records/DGE_numerical_and_scene_adaptation.patch').write_text(patch)
print('Clipped signed camera angle input to mathematical domain; all other algorithm settings unchanged')
