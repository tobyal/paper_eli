"""Three-panel layout; preserve the original scene, trajectory and data table."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiment figure/Experimental_Setup'
HISTORY = OUT / '_history/20261114_before_sample020_selection'
DATA = OUT / 'data'
KEYS = [20, 92, 198, 262]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def backup():
    HISTORY.resolve().relative_to(OUT.resolve())
    assert not HISTORY.is_symlink()
    if HISTORY.exists():
        return
    HISTORY.mkdir(parents=True)
    files = [p for p in OUT.iterdir() if p.is_file()]
    for subdir in ('panels', 'code'):
        files += [p for p in (OUT / subdir).rglob('*') if p.is_file()]
    records = {}
    for src in files:
        rel = src.relative_to(OUT)
        dest = HISTORY / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        assert sha(src) == sha(dest)
        records[rel.as_posix()] = sha(src)
    save(HISTORY / 'backup_hashes.json', records)


def prepare_local():
    import numpy as np
    p = DATA / 'left_camera_review'
    prior = read(OUT / 'sources.json')
    poses = read(DATA / 'camera_poses.json')['cameras']
    xy = (np.array([r['center'] for r in poses]) - np.array(prior['trajectory']['projection_origin'])) @ np.array(prior['trajectory']['projection_basis_columns'])
    selections = []
    for expected in [20]:
        i = next(i for i, row in enumerate(poses) if row['sample_id'] == expected)
        n = poses[i]['sample_id']
        assert n == expected
        image = p / f'frame_{n:06d}.jpg'
        selections.append({'sample_id': n, 'selection_reason': 'User selected sample 020 after reviewing real frames left of sample 036; reduced foreground crate obstruction.',
                           'actual_display_position': xy[i].tolist(),
                           'nominal_sample_time_s': (n - 1) / 2,
                           'selected_rgb': image.relative_to(OUT).as_posix(), 'sha256': sha(image),
                           'original_sample_path': f'/mnt/sda/ubuntu/lkx/papers/paper_eli/data/industrial/input/frame_{n:06d}.jpg'})
    save(p / 'selection.json', {'selections': selections,
                              'basis': 'User-approved real sample 020; other selected cameras and temporal order preserved.',
                              'video': read(DATA / 'dataset_audit.json')['sources'][0]})
    backup()
    print(json.dumps({'backup': str(HISTORY), 'selected_samples': [s['sample_id'] for s in selections]}))


def render():
    import numpy as np
    from PIL import Image
    import figure_experimental_setup as style
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox
    plt = style.plt
    backup()
    prior = read(HISTORY / 'sources.json')
    poses = read(DATA / 'camera_poses.json')['cameras']
    by_id = {p['sample_id']: p for p in poses}
    indices = {p['sample_id']: i for i, p in enumerate(poses)}
    selection = read(DATA / 'left_camera_review/selection.json')
    paths = {n: DATA / 'sampled_rgb' / f'frame_{n:06d}.jpg' for n in KEYS}
    paths[262] = DATA / 'camera_reselection/frame_000262.jpg'
    for row in selection['selections']:
        paths[row['sample_id']] = OUT / row['selected_rgb']
        assert sha(paths[row['sample_id']]) == row['sha256'] == sha(Path(row['original_sample_path']))
    for n in [92, 198, 262]:
        assert sha(paths[n]) == next(p for p in prior['keyframe_sources'] if p['sample_id'] == n)['raw']['sha256']
    render_path = DATA / 'saved_renders/frame_000213.png'
    assert sha(render_path) == prior['original_3dgs_render']['sha256']
    origin = np.array(prior['trajectory']['projection_origin'])
    basis = np.array(prior['trajectory']['projection_basis_columns'])
    lo, hi = np.array(prior['trajectory']['display_limits'])
    C = np.array([p['center'] for p in poses]); F = np.array([p['forward'] for p in poses])
    xy = (C - origin) @ basis; fxy = F @ basis
    cloud = np.load(DATA / 'sparse_cloud_preview.npz')
    pxy = (cloud['xyz'] - origin) @ basis
    visible = np.all((pxy >= lo) & (pxy <= hi), axis=1)

    def trajectory(ax):
        ax.scatter(pxy[visible, 0], pxy[visible, 1], s=.35, c='#B3B8BE', alpha=.4,
                   edgecolors='none', rasterized=True, zorder=1)
        ax.plot(xy[:, 0], xy[:, 1], color=style.BLUE, lw=.95, alpha=.9, zorder=3)
        ax.scatter(xy[:, 0], xy[:, 1], color=style.BLUE, s=2.5, edgecolors='none', zorder=4)
        ids = np.array([indices[n] for n in [20, 92, 153, 198, 244, 262, 304, 350]])
        ax.quiver(xy[ids, 0], xy[ids, 1], fxy[ids, 0] * .6, fxy[ids, 1] * .6,
                  angles='xy', scale_units='xy', scale=1, color=style.TEAL, width=.006,
                  headwidth=3.6, headlength=4.4, zorder=5)
        offsets = {20: (5, -12), 92: (5, 5), 198: (5, 5), 262: (-27, -14)}
        for n in KEYS:
            i = indices[n]
            ax.scatter(*xy[i], s=20, facecolor=style.ORANGE, edgecolor='white', linewidth=.5, zorder=6)
            ax.annotate(f'{n:03d}', xy[i], xytext=offsets[n], textcoords='offset points',
                        fontsize=6.8, color='#333333', zorder=7)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel='u (SfM units)', ylabel='v (SfM units)')
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.5, pad=2)
        ax.grid(color='#EBECEE', linewidth=.5, zorder=0)
        ax.spines[['left', 'bottom']].set_color('#B7BDC3')
        ax.legend(handles=[Line2D([], [], color=style.BLUE, marker='.', ms=3, lw=.8, label='Cameras / frame-order path'),
                           Line2D([], [], color=style.TEAL, marker='>', ms=4, lw=0, label='Viewing direction')],
                  loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False, ncol=1,
                  handlelength=1.7, borderaxespad=0, labelspacing=.2)

    figure_height = 3.80
    fig = plt.figure(figsize=(7.0866, figure_height), facecolor='white')
    fig.text(.018, .974, '(a) Video Acquisition', fontsize=9, fontweight='bold', va='top')
    w, gap = .232, .012
    h = w * 7.0866 * 9 / 16 / figure_height
    for i, n in enumerate(KEYS):
        x = .018 + i * (w + gap)
        style.image_axis(fig, [x, .652, w, h], Image.open(paths[n]))
        fig.text(x + w / 2, .641, f'Sample {n:03d}', fontsize=7, ha='center', va='top')
    fig.text(.018, .572, '(b) COLMAP Camera Trajectory', fontsize=9, fontweight='bold', va='top')
    fig.text(.525, .572, '(c) Reconstructed 3DGS Scene', fontsize=9, fontweight='bold', va='top')
    trajectory(fig.add_axes([.074, .090, .398, .366]))
    style.image_axis(fig, [.525, .048, .457, .4795], Image.open(render_path))
    fig.text(.7535, .031, 'Original 3DGS; camera of sample 213', fontsize=7, ha='center', va='top', color='#444444')
    style.export(fig, OUT / 'figure_industrial_scene_overview')
    panels = OUT / 'panels'
    # Retire only obsolete frame panels; revised current panels may be regenerated.
    stale = {'a_sample_036.jpg'}
    for p in list(panels.iterdir()):
        if p.is_file() and p.name in stale and (HISTORY / 'panels' / p.name).exists():
            assert sha(p) == sha(HISTORY / 'panels' / p.name)
            p.resolve().relative_to(OUT.resolve())
            p.unlink()
    fig.savefig(panels / 'a_video_acquisition.png', dpi=300,
                bbox_inches=Bbox.from_extents(.018 * 7.0866, .599 * figure_height, .982 * 7.0866, .908 * figure_height))
    for n in KEYS: shutil.copy2(paths[n], panels / f'a_sample_{n:03d}.jpg')
    Image.open(render_path).save(panels / 'c_reconstructed_scene.png')
    tf = plt.figure(figsize=(3.5, 2.4), facecolor='white')
    trajectory(tf.add_axes([.14, .18, .83, .66]))
    style.export(tf, panels / 'b_camera_trajectory')
    plt.close(tf); plt.close(fig)
    prior.update({'layout': 'Top: four acquisition frames; bottom: trajectory and original 3DGS.',
                  'acquisition_sample_ids': KEYS, 'revision_date': '2026-11-14',
                  'removed_display_panel': 'Industrial Scene (previous panel a)',
                  'first_frame_adjustment': selection, 'table_unchanged': True,
                  'panel_sources': {'a': [str(paths[n].relative_to(OUT)) for n in KEYS],
                                    'b': 'Same full camera trajectory, projection basis and limits; sample 020 replaces highlighted sample 036. Samples 092, 198 and 262 remain unchanged.',
                                    'c': prior['original_3dgs_render']},
                  'keyframe_sources': [{'sample_id': n, 'raw': {'path': str(paths[n]), 'sha256': sha(paths[n])}} for n in KEYS]})
    prior.pop('late_frame_selection', None)
    save(OUT / 'sources.json', prior)
    camera_manifest = read(HISTORY / 'camera_manifest.json')
    camera_manifest['keyframes'] = [by_id[n] for n in KEYS]
    save(OUT / 'camera_manifest.json', camera_manifest)
    (OUT / 'caption.txt').write_text(
        'Fig. X. Overview of industrial video acquisition, camera trajectory, and the reconstructed workstation environment. '
        '(a) Four actual video samples in chronological order (020, 092, 198 and 262) show the acquisition coverage of industrial equipment and workstations. '
        'Labels are indices in the 2-fps sampled sequence, not native video-frame indices. '
        '(b) All 350 recovered COLMAP camera centers, connected in frame order, with selected viewing directions and sparse SfM points. '
        'Centers use C = -R^T t. The same best-fit-plane display and uncalibrated SfM coordinate units are retained; orange labels correspond to (a). '
        '(c) The unchanged original 3DGS reconstruction at the registered camera of sample 213. '
        'No generated equipment or reconfigured scene is shown.\n', encoding='utf-8')
    old_readme = (HISTORY / 'README.txt').read_text(encoding='utf-8')
    (OUT / 'README.txt').write_text(old_readme + '\n2026-11-14 camera selection revision: samples 020/092/198/262. '
                                  'Sample 020 was selected by the user after reviewing real frames left of sample 036; the foreground crate obstruction is reduced. '
                                  'Trajectory geometry, saved original 3DGS render and statistics table unchanged. '
                                  'Previous figure and panels: _history/20261114_before_sample020_selection/.\n', encoding='utf-8')
    shutil.copy2(Path(__file__), OUT / 'code' / Path(__file__).name)
    table_hashes = {p.name: sha(p) for p in OUT.glob('table*') if p.is_file()}
    assert all(sha(HISTORY / name) == digest for name, digest in table_hashes.items())
    print(json.dumps({'output': str(OUT), 'samples': KEYS, 'table_unchanged': True,
                      'trajectory_geometry_unchanged': True, 'render_unchanged': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-local', action='store_true')
    args = parser.parse_args()
    prepare_local() if args.prepare_local else render()
