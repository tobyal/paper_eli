"""Create an official-source DGE copy with runtime and audit-only patches."""
import argparse
import difflib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'
REPO = ROOT / 'repos/DGE'
CODE = BASE / 'vendor/DGE_native'


def prepare(case):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if not CODE.exists():
        CODE.mkdir(parents=True)
        archive = subprocess.check_output(['git', 'archive', commit], cwd=REPO)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(CODE)
        # Keep documented runtime fixes, but restore the exact official segmentor.
        for name in ['gaussiansplatting/gaussian_renderer/__init__.py',
                     'threestudio/models/prompt_processors/stable_diffusion_prompt_processor.py',
                     'threestudio/systems/DGE.py']:
            text = (REPO / name).read_text()
            if name.endswith('/DGE.py'):
                text = text.replace('self.cfg.seg_prompt, camera=cur_cam)', 'self.cfg.seg_prompt)')
                text = text.replace('        self.text_segmentor = LangSAMTextSegmentor().to(get_device())',
                                    '        self.text_segmentor = LangSAMTextSegmentor().to(get_device())\n        import native_audit\n        native_audit.install(self)')
                text = text.replace('        self.gaussian.apply_grad_mask(selected_mask)',
                                    '        self.gaussian.apply_grad_mask(selected_mask)\n        import native_audit\n        native_audit.localization(self, selected_mask)')
                text = text.replace('            weights /= weights_cnt + 1e-7',
                                    '            weights /= weights_cnt + 1e-7\n            import native_audit\n            native_audit.weights(weights, weights_cnt)')
                text = text.replace('    def sort_the_cameras_idx(self, cams):',
                                    '        import native_audit\n        native_audit.edited(self)\n\n    def sort_the_cameras_idx(self, cams):')
                text = text.replace('    def training_step(self, batch, batch_idx):',
                                    '    def training_step(self, batch, batch_idx):\n        import native_audit\n        native_audit.progress(self)')
                text = text.replace('        self.gaussian.save_ply(save_path)',
                                    '        self.gaussian.save_ply(save_path)\n        import native_audit\n        native_audit.final(self, save_path)')
            (CODE / name).write_text(text)
    out = BASE / 'dge_native' / case
    if (out / 'config.json').exists():
        return
    for name in ['records', 'logs', 'localization', 'edited', 'optimization']:
        (out / name).mkdir(parents=True, exist_ok=True)
    row = next(r for r in json.loads((BASE / 'prompts.json').read_text())['cases'] if r['case'] == case)
    prompts = {key: row[key] for key in ['A', 'B']}
    prompts['edit_prompt'] = 'Replace ' + row['A'] + ' with ' + row['B']
    cfg = dict(case=case, upstream_commit=commit, source_scene=str(ROOT / 'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'),
               data_source=str(ROOT / 'data/industrial'), seed=0, mask_threshold=0.8,
               sampled_views=20, total_original_views=350, iterations=1000,
               camera_update_per_step=500, instance_rule='Official first LangSAM candidate; no ROI ranking',
               input_allowlist=['original 3DGS', 'original full COLMAP', 'A', 'B'])
    for name, data in [('config.json', cfg), ('prompts.json', prompts)]:
        (out / name).write_text(json.dumps(data, indent=2))
    patch = []
    for name in ['gaussiansplatting/gaussian_renderer/__init__.py', 'threestudio/systems/DGE.py',
                 'threestudio/models/prompt_processors/stable_diffusion_prompt_processor.py']:
        original = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=REPO, text=True)
        patch.extend(difflib.unified_diff(original.splitlines(True), (CODE / name).read_text().splitlines(True),
                                         fromfile='official/' + name, tofile='native/' + name))
    (out / 'records/compatibility_and_exports.patch').write_text(''.join(patch))
    (out / 'records/upstream_commit.txt').write_text(commit + '\n')
    print('Prepared', case, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case'); prepare(p.parse_args().case)
