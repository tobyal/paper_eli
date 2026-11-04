"""Optional industrial-scene adaptation; preserve all official core algorithms."""
import os
from pathlib import Path
import subprocess
import shutil

root=Path(os.environ['ROOT'])
def replace(file,before,after):
    text=file.read_text()
    if after not in text:
        assert text.count(before)==1, (str(file),before)
        file.write_text(text.replace(before,after))

for repo_name,system_name in [('GaussianEditor','GassuianEditor.py'),('DGE','DGE.py'),('DreamCatalyst/threestudio','GassuianEditor.py')]:
    repo=root/'repos'/repo_name
    shutil.copy2(root/'scripts/instance_selection.py',repo/'instance_selection.py')
    wrapper=repo/'threestudio/utils/sam.py'
    replace(wrapper,'from lang_sam import LangSAM','from lang_sam import LangSAM\nfrom instance_selection import select_instance')
    replace(wrapper,'def forward(self, images, prompt: str):','def forward(self, images, prompt: str, camera=None):')
    replace(wrapper,'            if mask.ndim == 3:',
            '            if camera is not None:\n                masks.append(select_instance(mask, camera))\n                continue\n            if mask.ndim == 3:')
    system=repo/'threestudio/systems'/system_name
    replace(system,'self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)',
            'self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt, camera=cur_cam)')
    patch=subprocess.check_output(['git','-C',str(repo),'diff','--','threestudio/utils/sam.py','threestudio/systems/'+system_name],text=True)
    (root/'records'/f'{repo_name.split("/")[0]}_instance_selection.patch').write_text(patch)

repo=root/'repos/cvpr25_EditSplat'
shutil.copy2(root/'scripts/instance_selection.py',repo/'instance_selection.py')
file=repo/'run_editing.py'
replace(file,'from lang_sam import LangSAM','from lang_sam import LangSAM\nfrom instance_selection import select_instance')
replace(file,'                mask, _, _, _ = lang_sam.predict(dst_image_pil, ed.target_mask_prompt)',
        '                mask, _, _, _ = lang_sam.predict(dst_image_pil, ed.target_mask_prompt)\n                mask = select_instance(mask, dst_camera)')
(root/'records/editsplat_instance_and_paths.patch').write_text(subprocess.check_output(['git','-C',str(repo),'diff','--','run_editing.py'],text=True))
print('Instance selection patches saved; diffusion, semantic tracing, anchor losses, attention, MFG and optimizers unchanged.')
