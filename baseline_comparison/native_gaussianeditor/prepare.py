"""Prepare an isolated native GE run without importing Ours spatial inputs."""
import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'
CODE = BASE / 'vendor/GaussianEditor_native'
UPSTREAM = '9249f847c57036266a0bfb3a210681f94264ede9'


def prepare(case):
    if not CODE.exists():
        shutil.copytree(BASE / 'vendor/GaussianEditor', CODE, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'ui_result', 'tmp_add'))
        sam = subprocess.check_output(['git', 'show', UPSTREAM + ':threestudio/utils/sam.py'],
                                      cwd=ROOT / 'repos/GaussianEditor')
        (CODE / 'threestudio/utils/sam.py').write_bytes(sam)
        (CODE / 'instance_selection.py').unlink()
        web = CODE / 'webui.py'
        text = web.read_text().replace('self.text_segmentor(this_frame, text_prompt, camera=cur_cam)',
                                       'self.text_segmentor(this_frame, text_prompt)')
        start = text.index('        if os.environ.get("GE_TARGET_CAMERA_DEPTH"):')
        end = text.index('        self.depth_scaler.visible = True', start)
        text = text[:start] + '''        # Native DPT depth and GUI Depth Scale; no external target geometry.
        import json
        Path(cache_dir, "placement.json").write_text(json.dumps({
            "native_z":float(z_in_cam), "depth_slider":float(self.depth_scaler.value),
            "source":"native DPT alignment and unmodified GUI depth-scale control",
            "camera_R":cam.R.tolist(), "camera_T":cam.T.tolist()},indent=2))
''' + text[end:]
        web.write_text(text)
        # Auditing-only snapshot of the exact semantic-tracing weights.
        needle = '        weights /= weights_cnt + 1e-7\n'
        assert needle in text
        text = text.replace(needle, needle + '''        if os.environ.get("GE_NATIVE_TRACE_DIR"):
            trace_dir = Path(os.environ["GE_NATIVE_TRACE_DIR"])
            trace_dir.mkdir(parents=True, exist_ok=True)
            np.save(trace_dir / "weights.npy", weights.detach().cpu().numpy())
            np.save(trace_dir / "observation_counts.npy", weights_cnt.detach().cpu().numpy())
''', 1)
        web.write_text(text)
    out = BASE / 'gaussianeditor_native' / case
    out.mkdir(parents=True, exist_ok=True)
    for name in ['records', 'logs', 'localization', 'add']:
        (out / name).mkdir(exist_ok=True)
    if (out / 'config.json').exists():
        print('Existing independent config preserved:', out)
        return
    spec = json.loads((BASE / 'prompts.json').read_text())
    row = next(x for x in spec['cases'] if x['case'] == case)
    # This allowlist deliberately omits all Ours cameras, boxes and instance ROIs.
    prompts = {k: row[k] for k in ['A', 'B', 'background_text', 'gaussianeditor_refine_text']}
    config = dict(case=case, source_scene=str(ROOT / 'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'),
                  colmap_source=str(ROOT / 'data/industrial'), upstream_commit=UPSTREAM,
                  seed=42, resolution=512, segmentation_views=24, segmentation_threshold=0.7,
                  delete_views=24, delete_steps=1500, delete_densification_interval=50,
                  add_margin_fraction=0.08, visible_mask_threshold=0.1, border_fraction=0.02,
                  depth_scale=1.0, depth_scale_policy='native GUI default; no external depth alignment',
                  instance_policy='upstream LangSAM first returned candidate, native multiview semantic tracing',
                  post_insertion_refinement=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    patches = []
    for name in ['webui.py', 'threestudio/utils/sam.py']:
        old = (BASE / 'vendor/GaussianEditor' / name).read_text()
        new = (CODE / name).read_text()
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                            fromfile='previous/' + name, tofile='native/' + name))
    (out / 'records/native_isolation.patch').write_text(''.join(patches))
    hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in CODE.rglob('*.py') if '__pycache__' not in p.parts}
    (out / 'records/source_sha256.json').write_text(json.dumps(hashes, indent=2))
    print('Prepared native-only inputs:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case', choices=['case00', 'case02', 'case04', 'case07', 'case10'])
    prepare(p.parse_args().case)
