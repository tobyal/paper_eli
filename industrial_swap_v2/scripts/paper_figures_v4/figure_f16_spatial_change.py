"""Read-only F16: visible projected occupancy before/after equipment replacement.

CUDA_VISIBLE_DEVICES=1 envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f16_spatial_change.py --render
envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f16_spatial_change.py --assemble
No localization, generation, registration, training, or Gaussian writes.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F16_spatial_change_analysis'
CASES = ['case00', 'case02', 'case07']
LABELS = ['Case00 | Six-axis Robot', 'Case02 | Roller Conveyor', 'Case07 | Storage Cabinet']
THRESHOLD = .25
RESOLUTION = 1536
COLORS = {'old_only': '#DF8A39', 'shared': '#9456AE', 'new_only': '#3489C9'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source(path):
    p = Path(path)
    return {'path': str(p), 'resolved': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def csvout(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rgb8(value):
    return np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)


def mask_image(path, mask):
    Image.fromarray(np.uint8(mask) * 255).save(path)


def bounds(mask):
    y, x = np.nonzero(mask)
    assert x.size, 'Empty visible target'
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def common_crop(mask):
    """3:2 integer crop containing every union pixel, plus local context."""
    x0, y0, x1, y1 = bounds(mask)
    h, w = mask.shape
    unit = math.ceil(max((x1 - x0) / (3 * .80), (y1 - y0) / (2 * .80)))
    unit = min(unit, w // 3, h // 2)
    cw, ch = unit * 3, unit * 2
    if cw < x1 - x0 or ch < y1 - y0:
        # Do not omit genuine selected projections to improve presentation.
        return [0, 0, w, h]
    left = min(max(round((x0 + x1 - cw) / 2), 0), w - cw)
    top = min(max(round((y0 + y1 - ch) / 2), 0), h - ch)
    assert left <= x0 and top <= y0 and left + cw >= x1 and top + ch >= y1
    return [left, top, left + cw, top + ch]


def case_paths(row):
    base = Path(row['ours_source'])
    if row['case'] == 'case07':
        placed = ROOT / 'industrial_swap_v2/outputs/case07_yaw_adjust_v1'
        scene = placed / 'scene_rotated_guarded.ply'
        composition = placed / 'composition_guarded.json'
    else:
        placed = base / 'm4'
        scene = placed / 'scene_final.ply'
        composition = placed / 'composition.json'
    return dict(base=base, scene=scene, composition=composition,
                obj=placed / 'object_world.ply', transform=placed / 'transform.json',
                target=base / 'm1/target_gaussians.ply',
                indices=base / 'm1/target_indices.npy',
                m1_manifest=base / 'm1/manifest.json',
                background=base / 'm1/background_gaussians.ply',
                guarded_background=base / 'm4/background_optimization/background_optimized.ply',
                guard=base / 'm4/background_optimization/summary.json',
                anchor=Path(row['ours_anchor']))


def render():
    import torch
    from plyfile import PlyData
    sys.path.insert(0, str(ROOT / 'gsle'))
    import common as c

    torch.cuda.set_per_process_memory_fraction(.35)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'render_sources.json').exists(), 'Existing raw results are preserved; use --assemble to relayout.'
    prompts_path = ROOT / 'baseline_comparison/prompts.json'
    cases = {r['case']: r for r in read(prompts_path)['cases']}
    original_path = ROOT / 'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    inputs = {'original_scene': source(original_path), 'case_index': source(prompts_path)}
    original = c.GaussianData.load(original_path)
    original_vertices = PlyData.read(str(original_path))['vertex'].data
    results = {}
    rows = []
    with torch.no_grad():
        for case in CASES:
            row = cases[case]
            paths = case_paths(row)
            folder = OUT / 'renders' / case
            folder.mkdir(parents=True, exist_ok=True)
            print('Rendering', case, flush=True)
            records = {key: source(p) for key, p in paths.items() if key != 'base'}
            m1 = read(paths['m1_manifest'])
            assert m1['source_sha256'] == inputs['original_scene']['sha256']
            ids = np.load(paths['indices']).astype(np.int64)
            assert len(np.unique(ids)) == len(ids) and ids.min() >= 0 and ids.max() < len(original.xyz)
            target_vertices = PlyData.read(str(paths['target']))['vertex'].data
            assert target_vertices.dtype == original_vertices.dtype
            assert np.array_equal(target_vertices, original_vertices[ids]), 'M1 PLY and source-index identity mismatch'
            del target_vertices
            anchor_record = read(paths['anchor'])
            camera = c.Camera.from_record(anchor_record)
            camera = camera.resize(RESOLUTION, round(camera.height * RESOLUTION / camera.width))
            camera_record = camera.record()
            dump(folder / 'camera.json', camera_record)

            original_rgb = c.render(camera, original, aux=False)['rgb'].cpu().numpy().transpose(1, 2, 0)
            flags = torch.zeros_like(original.xyz)
            flags[torch.as_tensor(ids, device='cuda')] = 1
            old_alpha = c.raster(camera, original, flags)[0].cpu().numpy()
            del flags
            old_object = original.subset(torch.as_tensor(ids, device='cuda'))
            old_independent_alpha = c.raster(camera, old_object, torch.ones_like(old_object.xyz))[0].cpu().numpy()
            del old_object

            composition = read(paths['composition'])
            assert records['scene']['sha256'] == composition['scene_sha256']
            assert records['obj']['sha256'] == composition['object_sha256']
            assert records['guarded_background']['sha256'] == composition['source_background_sha256']
            scene_vertices = PlyData.read(str(paths['scene']))['vertex'].data
            object_vertices = PlyData.read(str(paths['obj']))['vertex'].data
            nobject = len(object_vertices)
            assert len(scene_vertices) == composition['total_points']
            assert nobject == composition['new_object_points']
            assert object_vertices.dtype == scene_vertices.dtype
            assert np.array_equal(object_vertices, scene_vertices[-nobject:]), 'Object tail identity mismatch'
            del scene_vertices, object_vertices
            scene = c.GaussianData.load(paths['scene'])
            reconfigured_rgb = c.render(camera, scene, aux=False)['rgb'].cpu().numpy().transpose(1, 2, 0)
            flags = torch.zeros_like(scene.xyz)
            flags[-nobject:] = 1
            new_alpha = c.raster(camera, scene, flags)[0].cpu().numpy()
            del flags
            obj = c.GaussianData.load(paths['obj'])
            new_independent_alpha = c.raster(camera, obj, torch.ones_like(obj.xyz))[0].cpu().numpy()
            del obj, scene
            gc.collect()
            torch.cuda.empty_cache()

            # Both masks use the same threshold on real front-to-back contribution.
            old_mask = old_alpha > THRESHOLD
            new_mask = new_alpha > THRESHOLD
            union = old_mask | new_mask
            regions = {'old_only': old_mask & ~new_mask, 'shared': old_mask & new_mask,
                       'new_only': new_mask & ~old_mask}
            assert sum(int(v.sum()) for v in regions.values()) == int(union.sum())
            crop = common_crop(union)
            x0, y0, x1, y1 = crop
            assert int(union[y0:y1, x0:x1].sum()) == int(union.sum())
            Image.fromarray(rgb8(original_rgb)).save(folder / 'original.png')
            Image.fromarray(rgb8(reconfigured_rgb)).save(folder / 'reconfigured.png')
            np.savez_compressed(folder / 'visible_contributions.npz',
                                old_visible_alpha=old_alpha, new_visible_alpha=new_alpha,
                                old_independent_alpha=old_independent_alpha,
                                new_independent_alpha=new_independent_alpha)
            mask_image(folder / 'old_visible_mask.png', old_mask)
            mask_image(folder / 'new_visible_mask.png', new_mask)
            mask_image(folder / 'union_mask.png', union)
            for name, mask in regions.items():
                mask_image(folder / f'{name}_mask.png', mask)
            counts = {k + '_pixels': int(v.sum()) for k, v in regions.items()}
            percentages = {k + '_pct': 100. * v.sum() / union.sum() for k, v in regions.items()}
            guard = read(paths['guard'])
            guard_info = {'selected_step': guard.get('selected_step'), 'iterations': guard.get('iterations'),
                          'selection': guard.get('selection'),
                          'background_is_guarded_output': True,
                          'displayed_candidate_step2000': False}
            if case == 'case07':
                pose_report = paths['obj'].parent / 'report.json'
                pose = read(pose_report)
                records['yaw_adjustment_report'] = source(pose_report)
                guard_info.update(pose_adjusted_after_background_guard=True,
                                  delta_yaw_degrees=pose['delta_yaw_degrees'],
                                  new_pose_guard_evaluation=False,
                                  pose_note=pose['note'])
            else:
                guard_info['pose_adjusted_after_background_guard'] = False
            rows.append(dict(case=case, old_gaussians=len(ids), new_gaussians=nobject,
                             camera=camera.name, resolution_width=camera.width, resolution_height=camera.height,
                             contribution_threshold=THRESHOLD, old_visible_pixels=int(old_mask.sum()),
                             new_visible_pixels=int(new_mask.sum()), union_pixels=int(union.sum()),
                             **counts, **{k: float(v) for k, v in percentages.items()},
                             guarded_background_selected_step=guard.get('selected_step'),
                             metric_scope='Full camera; identical crop retains all union pixels'))
            # Check immutable inputs after rendering; no source PLY is ever saved.
            assert all(sha(Path(r['path'])) == r['sha256'] for r in records.values())
            results[case] = dict(sources=records, original_description=row['A'],
                                 replacement_description=row['B'], camera=camera_record,
                                 saved_anchor_camera=anchor_record, crop_xyxy=crop,
                                 union_pixels_retained_by_crop=int(union.sum()),
                                 guard_status=guard_info, old_target_identity_verified=True,
                                 new_object_tail_all_parameters_verified=True,
                                 input_hashes_unchanged=True, statistics=rows[-1])
            print(case, {k: round(float(v), 3) for k, v in percentages.items()}, 'crop', crop,
                  'guard step', guard.get('selected_step'), flush=True)
    del original, original_vertices
    assert sha(original_path) == inputs['original_scene']['sha256']
    csvout(OUT / 'spatial_change_metrics.csv', rows)
    dump(OUT / 'render_sources.json', dict(inputs=inputs, cases=results,
        visible_mask_definition='Target-tag contribution in the actual full scene, sum(T_i * alpha_i) over selected Gaussians, greater than 0.25; same threshold for both states.',
        occlusion_handling='All scene Gaussians participate in front-to-back rasterization. Old masks use the original scene; new masks use the displayed reconfigured scene.',
        independent_alpha_saved_for_audit_only=True, morphology_or_mask_cleanup=False,
        percentage_denominator='old_visible_mask OR new_visible_mask, all pixels in the camera image',
        interpretation='Two-dimensional visible projected occupancy, not physical 3D volume or ground-truth segmentation.',
        no_training_no_registration_no_gaussian_writes=True))


def assemble():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    records = read(OUT / 'render_sources.json')
    (OUT / 'panels').mkdir(exist_ok=True)
    (OUT / 'code').mkdir(exist_ok=True)
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    # 3600+ pixels, vector PDF labels; ~8 pt text when placed at 180 mm width.
    fig = plt.figure(figsize=(14.2, 11.8), facecolor='white')
    left, gap, width = .018, .012, (.964 - .024) / 3
    panel_height = width * 14.2 / 11.8 * (2 / 3)
    row_tops = [.918, .625, .332]
    titles = ['Original Equipment', 'Reconfigured Scene', 'Spatial Change Map']
    for j, title in enumerate(titles):
        fig.text(left + j * (width + gap) + width / 2, .982, title,
                 ha='center', va='top', weight='bold', fontsize=19)
    panel_records = {}
    for i, case in enumerate(CASES):
        info = records['cases'][case]
        folder = OUT / 'renders' / case
        full_old = np.asarray(Image.open(folder / 'original.png').convert('RGB')) / 255.
        full_new = np.asarray(Image.open(folder / 'reconfigured.png').convert('RGB')) / 255.
        contribution = np.load(folder / 'visible_contributions.npz')
        old_alpha = contribution['old_visible_alpha']
        old_mask = np.asarray(Image.open(folder / 'old_visible_mask.png')) > 0
        new_mask = np.asarray(Image.open(folder / 'new_visible_mask.png')) > 0
        # The localization tint uses actual continuous transmittance, not a filled mask.
        tint = np.asarray(to_rgb(COLORS['old_only']))
        weight = .48 * np.clip(old_alpha, 0, 1)
        highlighted = full_old * (1 - weight[..., None]) + tint * weight[..., None]
        gray = full_old @ np.array([.2126, .7152, .0722])
        spatial = .91 + .09 * np.repeat(gray[..., None], 3, axis=2)
        regions = {'old_only': old_mask & ~new_mask, 'shared': old_mask & new_mask,
                   'new_only': new_mask & ~old_mask}
        for name, mask in regions.items():
            spatial[mask] = to_rgb(COLORS[name])
        x0, y0, x1, y1 = info['crop_xyxy']
        images = [highlighted[y0:y1, x0:x1], full_new[y0:y1, x0:x1], spatial[y0:y1, x0:x1]]
        fig.text(left, row_tops[i] + .008, LABELS[i], fontsize=16, weight='bold', va='bottom')
        paths = []
        for j, (name, image) in enumerate(zip(['original_equipment', 'reconfigured_scene', 'spatial_change'], images)):
            path = OUT / 'panels' / f'{case}_{name}.png'
            Image.fromarray(rgb8(image)).save(path, dpi=(300, 300))
            paths.append(source(path))
            ax = fig.add_axes([left + j * (width + gap), row_tops[i] - panel_height, width, panel_height])
            ax.imshow(image, interpolation='nearest')
            ax.set_axis_off()
        metric = info['statistics']
        for k, region in enumerate(['old_only', 'shared', 'new_only']):
            fig.text(left + 2 * (width + gap) + width * (k + .5) / 3,
                     row_tops[i] - panel_height - .008,
                     f"{metric[region + '_pct']:.2f}%", color=COLORS[region],
                     ha='center', va='top', fontsize=17, weight='medium')
        panel_records[case] = paths
    fig.legend(handles=[Patch(facecolor=COLORS[k], label=label) for k, label in
                        [('old_only', 'Old-only region'), ('shared', 'Shared region'), ('new_only', 'New-only region')]],
               loc='lower center', bbox_to_anchor=(.5, .008), ncol=3, frameon=False,
               handlelength=1.3, columnspacing=2, fontsize=16)
    fig.savefig(OUT / 'figure_F16.png', dpi=300, facecolor='white')
    fig.savefig(OUT / 'figure_F16.pdf', facecolor='white')
    plt.close(fig)
    caption = (
        'Fig. 16. Spatial change analysis of cross-category industrial equipment replacement. '
        'The rows show an industrial machine replaced by a six-axis robotic arm (Case00), a wheeled '
        'machine replaced by a roller conveyor (Case02), and a storage-crate region replaced by an '
        'automated storage cabinet (Case07). Left: the original 3DGS scene, with the actual M1-selected '
        'Gaussians highlighted using continuous orange visibility contribution. Middle: the existing '
        'registered equipment integrated into the scene. Right: old-only (orange), shared (purple), '
        'and new-only (blue) visible projected occupancy. Each row uses exactly the same saved Anchor '
        'camera, intrinsics, rasterizer, resolution, and crop in all three columns. Old and new masks '
        'are obtained by rendering target tags within the respective full scenes, including foreground '
        'occlusion, and thresholding accumulated target contribution at 0.25 for both. No mask '
        'morphology or hand correction is applied. Region percentages use the full-image mask union '
        'as their denominator; the displayed crop retains every union pixel. The color-coded '
        'percentages below each change map follow the old-only, shared, new-only order. These measurements '
        'describe two-dimensional visible projections, not physical three-dimensional volume. '
        'Case00 and Case02 use stored guarded scene_final outputs (the guard retained Step 0). '
        'Case07 uses the stored, user-confirmed yaw correction with the guarded Step-0 background; '
        'no new guard evaluation was performed after that rigid pose correction. The rejected '
        '2,000-step background candidates are not used. Original masks, Gaussian selections, '
        'transforms, and scene parameters remain unchanged; no M1-M4 stage was rerun.'
    )
    (OUT / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    dump(OUT / 'sources.json', dict(**records, panels=panel_records,
                                  presentation={'colors': COLORS, 'highlight_opacity': .48,
                                                'source_image_resampling': 'none before final figure compositing',
                                                'spatial_map_background': 'Original scene luminance faded to 91-100% white',
                                                'figure_inches': [14.2, 11.8], 'png_dpi': 300},
                                  script=source(Path(__file__))))
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1900', '-png',
                    str(OUT / 'figure_F16.pdf'), str(OUT / 'pdf_preview')], check=True)
    Image.open(OUT / 'pdf_preview.png').convert('RGB').save(OUT / 'preview.jpg', quality=94)
    (OUT / 'README.txt').write_text(
        'F16: visible projected spatial change, Case00/02/07.\n'
        'figure_F16.png / figure_F16.pdf: 300 DPI figure and vector-text PDF.\n'
        'panels/: nine independent image panels.\n'
        'renders/: full scenes, camera JSON, float32 contributions, raw binary masks.\n'
        'spatial_change_metrics.csv: actual pixel counts and union-relative percentages.\n'
        'sources.json / caption.txt: scene identity, guard provenance, mask and metric definitions.\n'
        'code/: reproducible read-only rendering and plotting script.\n'
        'Old failure-mode F16 and all experiment artifacts were preserved.\n', encoding='utf-8')
    print('F16 complete:', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    args = parser.parse_args()
    assert args.render or args.assemble
    if args.render:
        render()
    if args.assemble:
        assemble()
