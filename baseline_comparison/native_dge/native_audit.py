"""Read-only exports around official DGE methods; no training/input changes."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

OUT = Path(os.environ['DGE_NATIVE_OUTPUT'])


def write(name, data):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def array(x):
    if isinstance(x, torch.Tensor): x = x.detach().float().cpu().numpy()
    if x.ndim == 4: x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4): x = x.transpose(1, 2, 0)
    if x.ndim == 3 and x.shape[-1] == 1: x = x[..., 0]
    return np.uint8(np.clip(x, 0, 1) * 255 + 0.5)


def save(name, x):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array(x)).save(path)


def montage(name, paths, labels=None, cols=3):
    canvas = Image.new('RGB', (cols * 512, ((len(paths) + cols - 1) // cols) * 540), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        x,y=i%cols*512,i//cols*540
        draw.text((x+8,y+7), labels[i] if labels else str(path), fill='black')
        canvas.paste(ImageOps.contain(Image.open(OUT/path).convert('RGB'),(512,512)),(x,y+28))
    canvas.save(OUT/name)


def camera_record(c, index):
    return dict(index=int(index), name=c.image_name, R=np.asarray(c.R).tolist(), T=np.asarray(c.T).tolist(),
                FoVx=float(c.FoVx), FoVy=float(c.FoVy), width=int(c.image_width),height=int(c.image_height))


def install(system):
    original=system.text_segmentor.model.predict; counter=[0]
    def predict(image, prompt, *args, **kwargs):
        result=original(image,prompt,*args,**kwargs)
        masks,boxes,phrases,logits=result; i=counter[0];counter[0]+=1
        folder=OUT/'localization'/'candidates'/f'{i:03d}';folder.mkdir(parents=True,exist_ok=True)
        image.save(folder/'input.png')
        np.savez_compressed(folder/'masks.npz',masks=masks.detach().cpu().numpy())
        write(str((folder/'metadata.json').relative_to(OUT)),dict(prompt=prompt,boxes=boxes.detach().cpu().tolist(),
              phrases=phrases,logits=logits.detach().cpu().tolist(),selected_index=0 if len(masks) else None))
        return result
    system.text_segmentor.model.predict=predict


def weights(w, counts):
    np.savez_compressed(OUT/'localization/tracing.npz',weights=w.cpu().numpy(),counts=counts.cpu().numpy())


@torch.no_grad()
def localization(s, mask):
    from gaussiansplatting.gaussian_renderer import render
    ids=np.flatnonzero(mask.cpu().numpy());np.save(OUT/'target_indices.npy',ids)
    cams=s.trainer.datamodule.train_dataset.scene.cameras
    ranks=[]
    for i in s.view_list:
        c=cams[i]; rgb=s.origin_frames[i][0]; semantic=render(c,s.gaussian,s.pipe,s.background_tensor,
                      override_color=mask[:,None].float().repeat(1,3))['render']
        m=torch.norm(semantic,dim=0)>0.8
        overlay=rgb.clone();overlay[m]=overlay[m]*0.4+rgb.new_tensor([1.,0.,0.])*0.6
        save(f'localization/view_{i:04d}_original.png',rgb)
        save(f'localization/view_{i:04d}_semantic.png',m.float())
        save(f'localization/view_{i:04d}_overlay.png',overlay)
        ranks.append((int(m.sum()),int(i)))
    ranks.sort(reverse=True);s.native_audit_views=[x[1] for x in ranks[:3]]
    i=s.native_audit_views[0];c=cams[i]
    shutil.copy2(OUT/f'localization/view_{i:04d}_original.png',OUT/'original_view.png')
    shutil.copy2(OUT/'original_view.png',OUT/'gaussian_optimization_initial.png')
    shutil.copy2(OUT/f'localization/view_{i:04d}_semantic.png',OUT/'localization_3d_mask.png')
    shutil.copy2(OUT/f'localization/view_{i:04d}_overlay.png',OUT/'localization_full_scene.png')
    mask_dir=Path(s.cache_dir)/(s.cfg.seg_prompt+f'_mask_{s.view_num}_view')
    shutil.copy2(mask_dir/f'{i:04d}.png',OUT/'localization_2d_mask.png')
    shutil.copy2(mask_dir/f'viz_{i:04d}.png',OUT/'localization_2d_overlay.png')
    s.gaussian.localize=True
    try:save('localization_selected_gaussians.png',render(c,s.gaussian,s.pipe,s.background_tensor)['render'])
    finally:s.gaussian.localize=False
    s.native_crop=Image.open(OUT/'localization_3d_mask.png').getbbox() or (0,0,512,512)
    montage('localization_multiview.png',[f'localization/view_{i:04d}_overlay.png' for _,i in ranks[:9]])
    montage('original_multiview.png',[f'localization/view_{i:04d}_original.png' for i in s.native_audit_views])
    write('records/localization.json',dict(selected_gaussians=len(ids),threshold=float(s.cfg.mask_thres),
          total_cameras=len(cams),sampled_cameras=[camera_record(cams[i],i) for i in s.view_list],
          review_cameras=[camera_record(cams[i],i) for i in s.native_audit_views],crop=list(s.native_crop),
          review_selection='Largest actual traced contribution among native sampled cameras; export only'))


@torch.no_grad()
def edited(s):
    step=int(s.true_global_step)
    for i,frame in s.edit_frames.items():save(f'edited/step_{step:04d}/{i:04d}.png',frame)
    view_ids=list(s.edit_frames)
    montage(f'edited/step_{step:04d}/overview.png',[f'edited/step_{step:04d}/{i:04d}.png' for i in view_ids],cols=5)
    shutil.copy2(OUT/f'edited/step_{step:04d}/overview.png',OUT/f'edited_2d_views_step{step}.png')
    if step==0:
        shutil.copy2(OUT/f'edited/step_{step:04d}/overview.png',OUT/'edited_2d_views.png')
        paths=[]
        for i in s.native_audit_views:
            paths.extend([f'localization/view_{i:04d}_original.png',f'edited/step_{step:04d}/{i:04d}.png'])
        montage('edited_2d_comparison.png',paths,cols=2)
    write(f'edited/step_{step:04d}/cameras.json',[camera_record(s.trainer.datamodule.train_dataset.scene.cameras[i],i) for i in view_ids])


@torch.no_grad()
def progress(s):
    step=int(s.true_global_step)
    if step!=500 or getattr(s,'native_middle_saved',False):return
    s.native_middle_saved=True
    from gaussiansplatting.gaussian_renderer import render
    c=s.trainer.datamodule.train_dataset.scene.cameras[s.native_audit_views[0]]
    save('gaussian_optimization_middle.png',render(c,s.gaussian,s.pipe,s.background_tensor)['render'])
    write('records/middle_iteration.json',dict(completed_optimizer_steps=step,source='Live Gaussian render before step 500; no reconstruction from final output'))


@torch.no_grad()
def final(s, path):
    from gaussiansplatting.gaussian_renderer import render
    for n,i in enumerate(s.native_audit_views):
        c=s.trainer.datamodule.train_dataset.scene.cameras[i]
        save(f'final_view_{n}.png',render(c,s.gaussian,s.pipe,s.background_tensor)['render'])
    shutil.copy2(OUT/'final_view_0.png',OUT/'scene_final.png')
    Image.open(OUT/'scene_final.png').crop(s.native_crop).save(OUT/'scene_final_crop.png')
    montage('final_multiview.png',[f'final_view_{i}.png' for i in range(3)])
    primary=s.native_audit_views[0]
    montage('final_comparison.png',['original_view.png',f'edited/step_0000/{primary:04d}.png','scene_final.png'],
            ['Original','First native 2D editing','Final Gaussian'])
    dst=OUT/'final_scene.ply'
    if not dst.exists():dst.symlink_to(Path(path).resolve())
    write('records/final.json',dict(iterations=int(s.true_global_step),gaussians=len(s.gaussian._xyz),source_ply=str(Path(path).resolve())))
