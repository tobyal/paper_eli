"""Allow official EditSplat to use verified local copies of the same models."""
import json
import os
from pathlib import Path
import subprocess

root=Path(os.environ['ROOT'])
repo=root/'repos/cvpr25_EditSplat'
file=repo/'run_editing.py'
text=file.read_text()
changes=[
 ('DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4",', 'DDIMScheduler.from_pretrained(os.environ.get("SD14_PATH", "CompVis/stable-diffusion-v1-4"),'),
 ('RM.load("ImageReward-v1.0")','RM.load(os.environ.get("IMAGEREWARD_PATH", "ImageReward-v1.0"), med_config=os.environ.get("IMAGEREWARD_CONFIG"))'),
 ('editsplat_Pipeline.from_pretrained("timbrooks/instruct-pix2pix",','editsplat_Pipeline.from_pretrained(os.environ.get("IP2P_PATH", "timbrooks/instruct-pix2pix"),')
]
for before,after in changes:
    if after not in text:
        assert text.count(before)==1, before
        text=text.replace(before,after)
file.write_text(text)
patch=subprocess.check_output(['git','-C',str(repo),'diff','--','run_editing.py'],text=True)
(root/'records/editsplat_local_model_paths.patch').write_text(patch)
print(patch)
