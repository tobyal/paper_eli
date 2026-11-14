"""Publication overview and table from audited video, SfM and original 3DGS data."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiment figure/Experimental_Setup'
DATA = OUT / 'data'
MAIN_ID = 213
KEY_IDS = [1, 92, 198, 350]
BLUE = '#225E9D'
TEAL = '#399276'
ORANGE = '#D57A27'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.titlesize': 9, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 6.8, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def image_axis(fig, rect, image):
    ax = fig.add_axes(rect)
    ax.imshow(image, interpolation='lanczos')
    ax.set_axis_off()
    return ax


def export(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')


def main():
    audit = read(DATA / 'dataset_audit.json')
    poses = read(DATA / 'camera_poses.json')['cameras']
    by_id = {row['sample_id']: row for row in poses}
    selected = {row['sample_id']: row for row in read(DATA / 'candidate_sources.json')}
    assert (audit['sampled_views'], audit['registered_views'], audit['training_views'], audit['heldout_views']) == (350, 350, 306, 44)
    for n in [MAIN_ID] + KEY_IDS:
        assert sha(DATA / 'sampled_rgb' / f'frame_{n:06d}.jpg') == selected[n]['raw']['sha256']
    assert sha(DATA / 'saved_renders' / f'frame_{MAIN_ID:06d}.png') == selected[MAIN_ID]['render']['sha256']
    panels = OUT / 'panels'
    panels.mkdir(exist_ok=True)
    raw = Image.open(DATA / 'sampled_rgb' / f'frame_{MAIN_ID:06d}.jpg').convert('RGB')
    render = Image.open(DATA / 'saved_renders' / f'frame_{MAIN_ID:06d}.png').convert('RGB')
    raw.save(panels / 'a_industrial_scene.png')
    render.save(panels / 'd_reconstructed_scene.png')
    for n in KEY_IDS:
        shutil.copy2(DATA / 'sampled_rgb' / f'frame_{n:06d}.jpg', panels / f'b_sample_{n:03d}.jpg')

    # Rigid display-plane projection only. No metric scale or fabricated route.
    C = np.array([row['center'] for row in poses])
    F = np.array([row['forward'] for row in poses])
    U = np.array([row['up'] for row in poses])
    origin = C.mean(0)
    _, singular_values, V = np.linalg.svd(C - origin, full_matrices=False)
    normal = V[2]
    if np.dot(normal, U.mean(0)) < 0: normal = -normal
    u = V[0]
    v = np.cross(normal, u)
    basis = np.stack([u, v], axis=1)
    xy = (C - origin) @ basis
    fxy = F @ basis
    points = np.load(DATA / 'sparse_cloud_preview.npz')
    pxy = (points['xyz'] - origin) @ basis
    lo, hi = xy.min(0), xy.max(0)
    span = hi - lo
    lo -= np.array([.09, .28]) * span
    hi += np.array([.09, .28]) * span
    point_visible = np.all((pxy >= lo) & (pxy <= hi), axis=1)
    indices = {row['sample_id']: i for i, row in enumerate(poses)}

    def trajectory(ax):
        ax.scatter(pxy[point_visible, 0], pxy[point_visible, 1], s=.35, c='#B3B8BE',
                   alpha=.4, edgecolors='none', rasterized=True, zorder=1)
        ax.plot(xy[:, 0], xy[:, 1], color=BLUE, lw=.95, alpha=.9, zorder=3)
        ax.scatter(xy[:, 0], xy[:, 1], color=BLUE, s=2.5, edgecolors='none', zorder=4)
        arrow_ids = [1, 47, 92, 153, 198, 244, 304, 350]
        ids = np.array([indices[n] for n in arrow_ids])
        ax.quiver(xy[ids, 0], xy[ids, 1], fxy[ids, 0] * .6, fxy[ids, 1] * .6,
                  angles='xy', scale_units='xy', scale=1, color=TEAL, width=.006,
                  headwidth=3.6, headlength=4.4, zorder=5)
        offsets = {1: (5, -12), 92: (5, 5), 198: (5, 5), 350: (-27, 7)}
        for n in KEY_IDS:
            i = indices[n]
            ax.scatter(*xy[i], s=20, facecolor=ORANGE, edgecolor='white', linewidth=.5, zorder=6)
            ax.annotate(f'{n:03d}', xy[i], xytext=offsets[n], textcoords='offset points',
                        fontsize=6.8, color='#333333', zorder=7)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
               xlabel='u (SfM units)', ylabel='v (SfM units)')
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.5, pad=2)
        ax.grid(color='#EBECEE', linewidth=.5, zorder=0)
        ax.spines[['left', 'bottom']].set_color('#B7BDC3')
        handles = [Line2D([], [], color=BLUE, marker='.', ms=3, lw=.8, label='Cameras / frame-order path'),
                   Line2D([], [], color=TEAL, marker='>', ms=4, lw=0, label='Viewing direction')]
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.01),
                  frameon=False, ncol=1, handlelength=1.7, borderaxespad=0, labelspacing=.2)

    # 180 mm wide; actual images retain their original aspect ratios and full FOV.
    fig = plt.figure(figsize=(7.0866, 4.72), facecolor='white')
    left, right, width = .018, .525, .457
    fig.text(left, .972, '(a) Industrial Scene', fontsize=9, fontweight='bold', va='top')
    fig.text(right, .972, '(b) Video Acquisition', fontsize=9, fontweight='bold', va='top')
    ax_a = image_axis(fig, [left, .532, width, .386], raw)
    fig.text(left + width / 2, .524, f'Sampled video frame {MAIN_ID:03d}', fontsize=7,
             ha='center', va='top', color='#444444')
    # Four genuine sampled frames in chronological order (not equal-spaced rendering poses).
    b_axes = []
    tile_w, tile_h = .2215, .1871
    for i, n in enumerate(KEY_IDS):
        x = right + (i % 2) * (tile_w + .014)
        y = .741 if i < 2 else .520
        ax = image_axis(fig, [x, y, tile_w, tile_h], Image.open(DATA / 'sampled_rgb' / f'frame_{n:06d}.jpg'))
        b_axes.append(ax)
        fig.text(x + tile_w / 2, y - .006, f'Sample {n:03d}', fontsize=7, ha='center', va='top')
    fig.text(left, .472, '(c) COLMAP Camera Trajectory', fontsize=9, fontweight='bold', va='top')
    fig.text(right, .472, '(d) Reconstructed 3DGS Scene', fontsize=9, fontweight='bold', va='top')
    ax_c = fig.add_axes([.072, .082, .398, .295])
    trajectory(ax_c)
    ax_d = image_axis(fig, [right, .042, width, .386], render)
    fig.text(right + width / 2, .030, f'Original 3DGS; camera of sample {MAIN_ID:03d}',
             fontsize=7, ha='center', va='top', color='#444444')
    export(fig, OUT / 'figure_industrial_scene_overview')
    # Editable standalone trajectory and acquisition montage, in addition to raw RGB panels.
    tf = plt.figure(figsize=(3.5, 2.4), facecolor='white')
    ta = tf.add_axes([.14, .18, .83, .66])
    trajectory(ta)
    export(tf, panels / 'c_camera_trajectory')
    plt.close(tf)
    fig.canvas.draw()
    from matplotlib.transforms import Bbox
    fig.savefig(panels / 'b_video_acquisition.png', dpi=300,
                bbox_inches=Bbox.from_extents(right * 7.0866, .486 * 4.72, .982 * 7.0866, .930 * 4.72))
    plt.close(fig)

    rows = [
        ('Scene type', 'Real industrial workstations in a manufacturing workshop'),
        ('Data source', 'One workshop video; frame sampling at 2 fps'),
        ('Video duration', f"{audit['video_duration_seconds']:.2f} s (2 min 54.90 s)"),
        ('Video resolution', f"{audit['video_width']} × {audit['video_height']} pixels"),
        ('Sampled / registered views', f"{audit['sampled_views']} / {audit['registered_views']} ({audit['training_views']} training views)"),
        ('Reconstruction method', 'COLMAP SfM + 3D Gaussian Splatting'),
        ('Held-out test views', f"{audit['heldout_views']} (every 8th registered view)"),
        ('Reconfiguration cases', '5: Case00, Case02, Case04, Case07 and Case10'),
    ]
    table_fig = plt.figure(figsize=(7.0866, 2.62), facecolor='white')
    table_ax = table_fig.add_axes([0, 0, 1, 1])
    table_ax.set_axis_off()
    table_ax.set_xlim(0, 1); table_ax.set_ylim(0, 1)
    table_ax.plot([.018, .982], [.965, .965], color='black', lw=1.0)
    table_ax.text(.03, .918, 'Item', fontweight='bold', fontsize=9, va='center')
    table_ax.text(.35, .918, 'Value', fontweight='bold', fontsize=9, va='center')
    table_ax.plot([.018, .982], [.872, .872], color='black', lw=.6)
    for i, (item, value) in enumerate(rows):
        y = .812 - i * .100
        table_ax.text(.03, y, item, fontsize=8.5, va='center')
        if i == 0:
            value = 'Real industrial workstations in a\nmanufacturing workshop'
        table_ax.text(.35, y, value, fontsize=8.5, va='center', linespacing=1.16)
    table_ax.plot([.018, .982], [.055, .055], color='black', lw=1.0)
    export(table_fig, OUT / 'table_industrial_scene_summary')
    plt.close(table_fig)
    table_caption = 'Table X. Summary of the industrial scene data and reconstruction setup.'
    (OUT / 'table_caption.txt').write_text(table_caption + '\n', encoding='utf-8')
    (OUT / 'table_industrial_scene_summary.md').write_text(table_caption + '\n\n| Item | Value |\n| --- | --- |\n'
        + '\n'.join(f'| {k} | {v} |' for k, v in rows) + '\n', encoding='utf-8')
    latex = '\\begin{table*}[t]\n\\centering\n\\caption{Summary of the industrial scene data and reconstruction setup.}\n'
    latex += '\\label{tab:industrial_scene_data}\n\\begin{tabular}{@{}p{0.29\\textwidth}p{0.66\\textwidth}@{}}\n\\toprule\nItem & Value \\\\\n\\midrule\n'
    for item, value in rows:
        latex += item + ' & ' + value.replace('×', '$\\times$') + ' \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    (OUT / 'table_industrial_scene_summary.tex').write_text('% Requires \\usepackage{booktabs}\n' + latex, encoding='utf-8')
    caption = (
        'Fig. X. Overview of the industrial scene, video acquisition, camera trajectory, and reconstructed workstation environment. '
        '(a) A representative RGB frame sampled from the real workshop video. '
        '(b) Four sampled frames in their original temporal order illustrate changes in viewing direction and scene coverage; '
        'the labels denote indices in the 2-fps extracted sequence, not native video frame indices. '
        '(c) All 350 recovered COLMAP camera centers, connected in sampled-frame order, with selected viewing directions and '
        'a lightly rendered subset of the sparse SfM points. Camera centers are computed as C = -R^T t. '
        'The plan-like display is an orthogonal projection onto the best-fit plane of the recovered camera centers; '
        'SfM coordinates are not metrically calibrated. Orange labels identify the four frames in (b). '
        '(d) The unedited original 3DGS reconstruction rendered using the registered camera of the same sampled frame as (a). '
        'The RGB frame retains the original video projection, whereas the saved 3DGS render uses COLMAP-undistorted pinhole intrinsics. '
        'No generated equipment or reconfigured scene is shown.'
    )
    (OUT / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    sources = {
        'purpose': 'Section 4.1 Experimental Setup; genuine acquisition and original reconstruction only',
        'paper_title': 'Language-Grounded 3D Gaussian Editing for Virtual Reconfiguration of Industrial Workstations',
        'audit': audit, 'main_sample_id': MAIN_ID, 'acquisition_sample_ids': KEY_IDS,
        'main_rgb_source': selected[MAIN_ID]['raw'], 'original_3dgs_render': selected[MAIN_ID]['render'],
        'main_camera': by_id[MAIN_ID],
        'main_rgb_dimensions': list(raw.size), 'saved_render_dimensions': list(render.size),
        'crop': 'Full image; no crop, no image-content modification',
        'trajectory': {'all_camera_count': len(C), 'formula': 'C = -R.T @ t; forward = R.T @ [0,0,1]',
                       'connection_order': [r['name'] for r in poses], 'projection_origin': origin.tolist(),
                       'projection_basis_columns': basis.tolist(), 'plane_normal': normal.tolist(),
                       'camera_center_singular_values': singular_values.tolist(),
                       'plane_rms_residual_sfm_units': float(np.sqrt(np.mean(((C-origin) @ normal)**2))),
                       'display_limits': [lo.tolist(), hi.tolist()], 'metric_units': False,
                       'sparse_points_in_source': audit['sparse_points'], 'sparse_points_sampled': len(pxy),
                       'sparse_points_in_display_extent': int(point_visible.sum())},
        'keyframe_sources': [selected[n] for n in KEY_IDS], 'table_rows': rows,
        'unknown_details': ['Acquisition device/model and mounting method were not recorded.',
                            'No physical-scale calibration is recorded.'],
        'reused_existing_renders': True, 'new_training_or_generation': False,
    }
    save(OUT / 'sources.json', sources)
    save(OUT / 'camera_manifest.json', {'main_view': by_id[MAIN_ID],
                                      'keyframes': [by_id[n] for n in KEY_IDS],
                                      'trajectory_poses': 'data/camera_poses.json',
                                      'projection': sources['trajectory']})
    (OUT / 'README.txt').write_text(
        'Section 4.1: industrial scene data overview\n\n'
        'figure_industrial_scene_overview: PNG (300 DPI), vector-text PDF and editable SVG.\n'
        'table_industrial_scene_summary: PNG, PDF, SVG, Markdown and LaTeX (booktabs).\n'
        'panels/: original RGB, four chronological sampled frames, trajectory, original 3DGS render.\n'
        'sources.json / camera_manifest.json / data/: actual source identities, all 350 poses and audit records.\n\n'
        'Video: 174.89655 s, 2560 x 1440 pixels, 5072 native frames, nominal 29 fps.\n'
        'Sampling: 2 fps, 350 frames at 1920 x 1080. All 350 registered.\n'
        'Undistorted calibrated images: 1875 x 1056; saved reconstruction renders: 938 x 528.\n'
        'Split: 306 training / 44 held-out (sorted zero-based index modulo 8 equals zero).\n'
        'Reconstruction: 30000 iterations; 1828777 original Gaussians; 136641 sparse SfM points.\n'
        'Sampling and split checked against actual files, binaries, reconstruction camera records and render counts.\n'
        'Acquisition hardware/mounting and physical-scale calibration are not recorded. No such claims are made.\n'
        'Neither the source scenes nor any previous confirmed figure has been changed.\n', encoding='utf-8')
    code = OUT / 'code'; code.mkdir(exist_ok=True)
    for filename in ('prepare_experimental_setup.py', 'figure_experimental_setup.py'):
        shutil.copy2(Path(__file__).parent / filename, code / filename)
    print(json.dumps({'output': str(OUT), 'main_sample': MAIN_ID, 'keyframes': KEY_IDS,
                      'camera_count': len(C), 'unknown_table_fields': []}))


if __name__ == '__main__': main()
