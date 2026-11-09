"""Resume native Add after its own completed image/mesh/Gaussian stages."""
from pathlib import Path
import difflib
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
p=BASE/'vendor/GaussianEditor/webui.py';old=p.read_text();new=old
needle='        frustum = None\n        frame = None\n        while True:\n            if self.inpaint_again:'
replacement='''        frustum = None
        frame = None
        cache_dir = Path(os.environ["GE_ADD_CACHE"]).absolute().as_posix()
        reuse = os.environ.get("GE_REUSE_ADD_ASSETS") == "1"
        if reuse:
            out = Image.open(Path(cache_dir) / "inpainted.png").convert("RGB")
            frame, frustum = ui_utils.new_frustum_from_cam(None, self.server, np.asarray(out))
            print("Reusing this attempt's native Add image and completed 3D stages", flush=True)
        while not reuse:
            if self.inpaint_again:'''
assert needle in new;new=new.replace(needle,replacement,1)
new=new.replace('        removed_bg = rembg.remove(out)','        removed_bg = Image.open(Path(cache_dir) / "removed_bg.png").convert("RGBA") if reuse else rembg.remove(out)',1)
new=new.replace('        out.save(inpaint_path)\n        removed_bg.save(removed_bg_path)','        if not reuse:\n            out.save(inpaint_path)\n            removed_bg.save(removed_bg_path)',1)
for idx,path in [(1,'mesh_path'),(2,'mesh_path'),(3,'gs_path')]:
 start=new.index(f'        p{idx} = subprocess.Popen(')
 end=new.index(f'        if p{idx}.wait() != 0: raise RuntimeError("Native Add subprocess {idx} failed")',start)
 end=new.index('\n',end)
 block=new[start:end]
 new=new[:start]+f'        if not (reuse and Path({path}).is_file()):\n'+'\n'.join('    '+l for l in block.split('\n'))+new[end:]
new=new.replace('if not (0 < requested_scale <= 5):','if not (math.isfinite(requested_scale) and requested_scale > 0):',1)
new=new.replace('"depth_slider":requested_scale,','"depth_slider":requested_scale,"native_ui_range":[0,5],"ui_range_extended_for_scene_units":requested_scale>5,',1)
assert new!=old;p.write_text(new)
(BASE/'records/gaussianeditor_native_resume_and_slider_range.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/webui.py',tofile='after/webui.py')))
print('Native Add cache resume; positive depth slider range adapted to scene units')
