"""Keep the original held-out cameras after filtering editing views."""
import os,subprocess,sys
from pathlib import Path
root=Path(os.environ['ROOT'])
repo=root/'repos/cvpr25_EditSplat'
path=repo/'scene/dataset_readers.py'
text=path.read_text()
old='''    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
'''
new='''    split_path = os.path.join(path, "split.json")
    if eval and os.path.isfile(split_path):
        with open(split_path) as split_file:
            explicit_split = json.load(split_file)
        train_names = {os.path.splitext(name)[0] for name in explicit_split["train"]}
        test_names = {os.path.splitext(name)[0] for name in explicit_split["test"]}
        assert not train_names.intersection(test_names)
        assert {c.image_name for c in cam_infos} == train_names.union(test_names)
        train_cam_infos = [c for c in cam_infos if c.image_name in train_names]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
'''
if '--remove' in sys.argv:
    if new in text: path.write_text(text.replace(new,old,1))
    print('Restored official index%8 split; target-centered dataset remains diagnostic only')
    raise SystemExit(0)
if new not in text:
    assert old in text
    text=text.replace(old,new,1)
    path.write_text(text)
patch=subprocess.check_output(['git','diff','--','scene/dataset_readers.py'],cwd=repo,text=True)
(root/'records/EditSplat_explicit_split.patch').write_text(patch)
print('EditSplat explicit train/test manifest adapter installed')
