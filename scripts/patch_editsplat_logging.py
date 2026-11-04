"""Add disk diagnostics only; preserve every EditSplat tensor and update."""
import os,subprocess
from pathlib import Path
root=Path(os.environ['ROOT']);repo=root/'repos/cvpr25_EditSplat';path=repo/'run_editing.py'
text=path.read_text()
changes=[
('''                edited_image_pil_list_RM.append(edited_pil)
''','''                edited_image_pil_list_RM.append(edited_pil)
                diagnostic_dir = os.path.join(dataset.model_path, "initial_2d")
                os.makedirs(diagnostic_dir, exist_ok=True)
                edited_pil.save(os.path.join(diagnostic_dir, f"{idx:06d}.png"))
'''),
('''            is_top_selection = [rank <= top_count for rank in ranking]
''','''            is_top_selection = [rank <= top_count for rank in ranking]
            with open(os.path.join(dataset.model_path, "imagereward_ranking.json"), "w") as diagnostic_file:
                json.dump({"ranking": [int(r) for r in ranking], "rewards": [float(r) for r in rewards], "selected": [bool(r) for r in is_top_selection], "top_count": top_count, "filtering_ratio": ed.filtering_ratio, "camera_names": [c.image_name for c in camera_list]}, diagnostic_file, indent=2)
'''),
('''            edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())
''','''            edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())
            diagnostic_dir = os.path.join(dataset.model_path, "mfg_2d")
            os.makedirs(diagnostic_dir, exist_ok=True)
            topilimage(edited_image_MFG_list[-1]).save(os.path.join(diagnostic_dir, f"{idx:06d}.png"))
'''),
('''        selected_mask = attn_weights[:, 0]
''','''        selected_mask = attn_weights[:, 0]
        torch.save(selected_mask.detach().cpu(), os.path.join(dataset.model_path, "attention_gaussian_weights.pt"))
''')]
for before,after in changes:
 if after not in text:
  assert text.count(before)==1,before
  text=text.replace(before,after,1)
path.write_text(text)
(root/'records/EditSplat_logging_and_adaptation.patch').write_text(subprocess.check_output(['git','diff','--','run_editing.py'],cwd=repo,text=True))
print('Added actual2D diagnostics, ImageReward ranks and attention weights; algorithm unchanged')
