"""Snapshot official Delete/Add; parameterize caches and headless UI only."""
from pathlib import Path
import shutil,json,difflib,hashlib
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
src=ROOT/'repos/GaussianEditor';dst=BASE/'vendor/GaussianEditor'
assert not dst.exists()
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','edit_cache','tmp_add','ui_result','.cache','outputs','wandb'))
models=json.loads((ROOT/'records/core_model_downloads.json').read_text())
extra=json.loads((BASE/'records/gaussianeditor_add_models.json').read_text())
recovered=json.loads((BASE/'records/gaussianeditor_recovered_models.json').read_text())
diffs=[]
def change(name,old,new):
 p=dst/name;t=p.read_text();assert old in t,(name,old);u=t.replace(old,new);p.write_text(u)
 diffs.extend(difflib.unified_diff(t.splitlines(True),u.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
sd=models['stable-diffusion-v1-5/stable-diffusion-v1-5']['path']
for name in ['webui.py','GUI/DelGuidance.py','train_from_mesh.py']:
 change(name,'runwayml/stable-diffusion-v1-5',sd)
change('webui.py','lllyasviel/control_v11p_sd15_inpaint',extra['lllyasviel/control_v11p_sd15_inpaint']['path'])
change('webui.py','self.text_segmentor(this_frame, text_prompt)','self.text_segmentor(this_frame, text_prompt, camera=cur_cam)')
change('webui.py','Path("tmp_add")','Path(os.environ["GE_ADD_CACHE"])')
change('webui.py','f"{sys.prefix}/bin/accelerate launch','f"{sys.executable} -m accelerate.commands.launch')
for i in (1,2,3):change('webui.py',f'        p{i}.wait()',f'        if p{i}.wait() != 0: raise RuntimeError("Native Add subprocess {i} failed")')
# Native interactive depth slider: associate inserted object with old target depth.
# No silhouette/RGB fitting or Ours transforms are used.
change('webui.py','        self.depth_scaler.visible = True',
'''        if os.environ.get("GE_TARGET_CAMERA_DEPTH"):
            native_depth = float(z_in_cam)
            requested_scale = float(os.environ["GE_TARGET_CAMERA_DEPTH"]) / native_depth
            if not (0 < requested_scale <= 5):
                raise RuntimeError(f"Native depth slider cannot reach target: z={native_depth}, scale={requested_scale}")
            self.depth_scaler.value = requested_scale
            import json
            Path(cache_dir, "placement.json").write_text(json.dumps({"native_z":native_depth,"depth_slider":requested_scale,"target_camera_depth":float(os.environ["GE_TARGET_CAMERA_DEPTH"]),"camera_R":cam.R.tolist(),"camera_T":cam.T.tolist()},indent=2))
        self.depth_scaler.visible = True''')
change('webui.py','                self.gaussian.concat_gaussians(new_object_gaussian)',
 '                new_object_gaussian.save_ply(os.path.join(cache_dir, "object_world.ply"))\n                self.gaussian.concat_gaussians(new_object_gaussian)')
t=(dst/'train_from_mesh.py').read_text();start=t.index('    prompt_processor_1024 =');end=t.index('\n    if "n2n"',start)
change('train_from_mesh.py',t[start:end],'    # Unused SD2.1 prompt allocation removed; no loss consumes it.\n')
guidance='threestudio/models/guidance/instructpix2pix_guidance.py'
change(guidance,'timbrooks/instruct-pix2pix',models['timbrooks/instruct-pix2pix']['path'])
change(guidance,'CompVis/stable-diffusion-v1-4',models['CompVis/stable-diffusion-v1-4']['path'])
cfg='threestudio/utils/wonder3D/configs/mvdiffusion-joint-ortho-6views.yaml'
change(cfg,'lambdalabs/sd-image-variations-diffusers',recovered['lambda/sd-image-variations-diffusers']['path'])
change(cfg,"'./ckpts/'",repr(extra['camenduru/Wonder3D']['path']))
change(cfg,'dataloader_num_workers: 64','dataloader_num_workers: 8')
dpt=dst/'.cache/dpt';dpt.mkdir(parents=True);(dpt/'omnidata_dpt_depth_v2.ckpt').symlink_to(ROOT/'models/omnidata/omnidata_dpt_depth_v2.ckpt')
(BASE/'records/gaussianeditor_headless.patch').write_text(''.join(diffs))
(BASE/'records/gaussianeditor_snapshot.json').write_text(json.dumps({str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*.py')},indent=2))
print(dst)
