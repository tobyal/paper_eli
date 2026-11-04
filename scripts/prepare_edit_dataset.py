"""Create common camera-correct 512px edits and a fixed held-out split.

The editing directories contain camera-only COLMAP text models plus the original
sparse point cloud. Feature observations are unnecessary for pretrained 3DGS editing.
The untouched full COLMAP reconstruction remains in data/industrial.
"""
import importlib.util
import json
import os
from pathlib import Path
from PIL import Image

root = Path(os.environ['ROOT'])
spec = importlib.util.spec_from_file_location('colmap_loader', root/'repos/GaussianEditor/gaussiansplatting/scene/colmap_loader.py')
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
source = root/'data/industrial'
cameras = io.read_intrinsics_binary(str(source/'sparse/0/cameras.bin'))
images = io.read_extrinsics_binary(str(source/'sparse/0/images.bin'))
ordered = sorted(images.values(), key=lambda im: im.name)
split = {'train': [], 'test': []}
camera_lines = []
crop_specs = {}
for camera in cameras.values():
    assert camera.model == 'PINHOLE', camera.model
    fx, fy, cx, cy = camera.params
    side = 2 * min(cx, cy, camera.width-cx, camera.height-cy)
    scale = 512/side
    crop_specs[camera.id] = (side/512, 0, cx-side/2, 0, side/512, cy-side/2)
    camera_lines.append(f'{camera.id} PINHOLE 512 512 {fx*scale:.16g} {fy*scale:.16g} 256 256\n')

for subset in split:
    dest = root/f'data/edit_{subset}'
    (dest/'images').mkdir(parents=True, exist_ok=True)
    (dest/'sparse/0').mkdir(parents=True, exist_ok=True)
    (dest/'sparse/0/cameras.txt').write_text(''.join(camera_lines))
    selected = [im for i, im in enumerate(ordered) if ((i % 8 == 0) == (subset == 'test'))]
    lines = []
    for im in selected:
        split[subset].append(im.name)
        pose = ' '.join(f'{v:.17g}' for v in list(im.qvec)+list(im.tvec))
        lines.append(f'{im.id} {pose} {im.camera_id} {im.name}\n\n')
        image = Image.open(source/'images'/im.name).convert('RGB')
        image = image.transform((512,512), Image.Transform.AFFINE, crop_specs[im.camera_id], Image.Resampling.BICUBIC)
        image.save(dest/'images'/im.name, quality=98)
    (dest/'sparse/0/images.txt').write_text(''.join(lines))
    ply = dest/'sparse/0/points3D.ply'
    if not ply.exists():
        ply.symlink_to(source/'sparse/0/points3D.ply')

# EditSplat performs its own default index%8 split. Give it all cameras so its
# training set is exactly the same 306-camera pool supplied to other methods.
dest=root/'data/edit_all'
(dest/'images').mkdir(parents=True,exist_ok=True)
(dest/'sparse/0').mkdir(parents=True,exist_ok=True)
(dest/'sparse/0/cameras.txt').write_text(''.join(camera_lines))
lines=[]
for im in ordered:
    pose=' '.join(f'{v:.17g}' for v in list(im.qvec)+list(im.tvec))
    lines.append(f'{im.id} {pose} {im.camera_id} {im.name}\n\n')
    image_path=dest/'images'/im.name
    subset='test' if im.name in split['test'] else 'train'
    if not image_path.exists(): image_path.symlink_to(root/f'data/edit_{subset}/images'/im.name)
(dest/'sparse/0/images.txt').write_text(''.join(lines))
if not (dest/'sparse/0/points3D.ply').exists():
    (dest/'sparse/0/points3D.ply').symlink_to(source/'sparse/0/points3D.ply')

(root/'records/camera_split.json').write_text(json.dumps(split, indent=2))
(root/'records/edit_image_transform.json').write_text(json.dumps({'resolution':[512,512], 'affine_by_camera':crop_specs, 'description':'Crop about calibrated principal point, uniform scale, preserve camera extrinsics; test = sorted index % 8 == 0'}, indent=2))
print({name:len(values) for name,values in split.items()})
