"""Create calibrated object-centric views via pure-rotation image homographies.

Camera centers and the shared 3D scene stay fixed. No new pixels are synthesized:
only source-image rays inside the original field of view are resampled.
"""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

root=Path(os.environ['ROOT'])
spec=importlib.util.spec_from_file_location('colmap_io',root/'repos/gaussian-splatting/scene/colmap_loader.py')
io=importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
source=root/'data/industrial'
cameras=io.read_intrinsics_binary(str(source/'sparse/0/cameras.bin'))
images=io.read_extrinsics_binary(str(source/'sparse/0/images.bin'))
geometry=json.loads((root/'records/target_geometry.json').read_text())
center=np.array(geometry['center'])
corners=center+(np.array(list(itertools.product([-1,1],repeat=3)))*np.array(geometry['half_extents']))@np.array(geometry['basis'])
original_split=json.loads((root/'records/camera_split.json').read_text())
heldout=set(original_split['test'])
records=[]
rejected=[]
for im in sorted(images.values(),key=lambda image:image.name):
    mask=cv2.imread(str(root/'outputs/evaluation_masks/wide'/Path(im.name).with_suffix('.png')),cv2.IMREAD_GRAYSCALE)
    if int((mask>127).sum())<800:
        rejected.append({'frame':im.name,'reason':'target visible area below800 pixels in960px source render'})
        continue
    intr=cameras[im.camera_id]
    fx,fy,cx,cy=intr.params
    old_K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    old_R=io.qvec2rotmat(im.qvec)
    C=-old_R.T@im.tvec
    ray=old_R@(center-C)
    ray=ray/np.linalg.norm(ray)
    if ray[2]<=0: continue
    z=np.array([0.,0.,1.])
    cross=np.cross(ray,z)
    sine=np.linalg.norm(cross)
    angle=np.arctan2(sine,np.dot(ray,z))
    Q=Rotation.from_rotvec(cross/sine*angle).as_matrix() if sine>1e-12 else np.eye(3)
    new_R=Q@old_R
    new_t=-new_R@C
    xyz=corners@new_R.T+new_t
    if np.any(xyz[:,2]<=0): continue
    norm=xyz[:,:2]/xyz[:,2:3]
    span=(norm.max(axis=0)-norm.min(axis=0)).max()
    focal=256/span
    canvas=np.array([[0,0,1],[511,0,1],[511,511,1],[0,511,1]],dtype=float)
    valid=False
    for _ in range(30):
        new_K=np.array([[focal,0,256],[0,focal,256],[0,0,1.]])
        H=new_K@new_R@old_R.T@np.linalg.inv(old_K)
        old_pixels=canvas@np.linalg.inv(H).T
        old_pixels=old_pixels[:,:2]/old_pixels[:,2:3]
        valid=(old_pixels[:,0].min()>=3 and old_pixels[:,1].min()>=3 and old_pixels[:,0].max()<intr.width-3 and old_pixels[:,1].max()<intr.height-3)
        if valid: break
        focal*=1.04
    target_span=focal*span
    if not valid or target_span>435:
        rejected.append({'frame':im.name,'reason':'cannot fit both complete target and source field of view'})
        continue
    qxyzw=Rotation.from_matrix(new_R).as_quat()
    qvec=qxyzw[[3,0,1,2]]
    if qvec[0]<0: qvec=-qvec
    assert np.max(np.abs(io.qvec2rotmat(qvec)-new_R))<1e-7
    assert np.linalg.norm(-new_R.T@new_t-C)<1e-7
    projected_center=(new_K@(new_R@center+new_t))
    assert np.linalg.norm(projected_center[:2]/projected_center[2]-[256,256])<1e-7
    record={'frame':im.name,'image_id':int(im.id),'camera_id':int(im.id),'qvec':qvec.tolist(),'tvec':new_t.tolist(),'focal':float(focal),'homography':H.tolist(),'source_camera_center':C.tolist(),'target_span_px':float(target_span),'split':'test' if im.name in heldout else 'train'}
    records.append(record)

split={subset:[r['frame'] for r in records if r['split']==subset] for subset in ['train','test']}
assert len(split['train'])>=40,(len(split['train']),len(split['test']))
for subset in ['train','test','all']:
    destination=root/f'data/edit_{subset}_target'
    (destination/'images').mkdir(parents=True,exist_ok=True)
    (destination/'sparse/0').mkdir(parents=True,exist_ok=True)
    chosen=[r for r in records if subset=='all' or r['split']==subset]
    camera_lines=[]
    image_lines=[]
    for r in chosen:
        camera_lines.append(f'{r["camera_id"]} PINHOLE 512 512 {r["focal"]:.17g} {r["focal"]:.17g} 256 256\n')
        pose=' '.join(f'{v:.17g}' for v in r['qvec']+r['tvec'])
        image_lines.append(f'{r["image_id"]} {pose} {r["camera_id"]} {r["frame"]}\n\n')
        path=destination/'images'/r['frame']
        if subset=='all':
            if not path.exists(): path.symlink_to(root/f'data/edit_{r["split"]}_target/images'/r['frame'])
        else:
            image=cv2.imread(str(source/'images'/r['frame']))
            warped=cv2.warpPerspective(image,np.array(r['homography']),(512,512),flags=cv2.INTER_CUBIC)
            cv2.imwrite(str(path),warped,[cv2.IMWRITE_JPEG_QUALITY,98])
    (destination/'sparse/0/cameras.txt').write_text(''.join(camera_lines))
    (destination/'sparse/0/images.txt').write_text(''.join(image_lines))
    cloud=destination/'sparse/0/points3D.ply'
    if not cloud.exists(): cloud.symlink_to(source/'sparse/0/points3D.ply')
    (destination/'split.json').write_text(json.dumps(split,indent=2))
manifest={'description':'Calibrated pure-rotation virtual cameras centered on the same physical machine; all output pixels map inside real source images. Target occupies50% of image maximum span unless zoom is needed to keep source pixels valid. Shared full-scene reconstruction unchanged. Original held-out44 cameras remain excluded from editing and remain the evaluation set.','split':split,'views':records,'rejected':rejected}
(root/'records/target_view_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'train':len(split['train']),'test':len(split['test']),'rejected':len(rejected)},indent=2))
