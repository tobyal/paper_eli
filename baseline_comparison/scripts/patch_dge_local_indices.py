"""Keep native local rendering but scatter its raster statistics to scene IDs."""
from pathlib import Path
import difflib,json,shutil,time
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
p=ROOT/'repos/DGE/gaussiansplatting/gaussian_renderer/__init__.py';old=p.read_text();new=old
new=new.replace('pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"',
 '(pc._xyz if getattr(pc, "localize", False) else pc.get_xyz), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"',1)
new=new.replace('means2D = screenspace_points','means2D = screenspace_points[pc.mask] if getattr(pc, "localize", False) else screenspace_points',1)
new=new.replace('colors_precomp = override_color','colors_precomp = override_color[pc.mask] if getattr(pc, "localize", False) else override_color',1)
needle='    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.'
new=new.replace(needle,'''    # Local rendering uses a subset; densification statistics use global IDs.
    if getattr(pc, "localize", False):
        global_radii = radii.new_zeros(pc._xyz.shape[0])
        global_radii[pc.mask] = radii
        radii = global_radii

'''+needle,1)
assert new!=old and 'global_radii' not in old
(BASE/'records/dge_local_indices.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/gaussian_renderer/__init__.py',tofile='after/gaussian_renderer/__init__.py')))
p.write_text(new)
# Preserve the real failed attempt before retrying with the compatibility fix.
out=BASE/'dge/case00';dest=out/'attempts/02_local_raster_index_error';dest.mkdir()
for name in ['official','edit_cache','logs/train.log','records/status.json','records/instance_selection.jsonl']:
 src=out/name
 if src.exists():shutil.move(str(src),str(dest/src.name))
print('patched native local raster/global densification index compatibility')
