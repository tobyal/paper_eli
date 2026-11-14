"""Relayout F14 from saved renders and logs only; no rasterizer or training imports."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F14_occlusion_refinement_revised'
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F14_occlusion_refinement_insets_flush'
COLORS = {'RGB-only': '#D48938', 'Full M4 Candidate': '#217D99'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (SOURCE / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def image(ax, name):
    pixels = np.asarray(Image.open(SOURCE / 'panels' / name))
    if pixels.ndim == 2:
        ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    else:
        ax.imshow(pixels)
    ax.set_axis_off()
    return pixels


def curve(ax, data, key, multiplier):
    for name, color in COLORS.items():
        series = [r for r in data if r['method'] == name]
        steps = [int(r['step']) for r in series]
        assert steps == list(range(0, 2001, 50))
        ax.plot(steps, [float(r[key]) * multiplier for r in series],
                color=color, lw=2.15, label=name)
    ax.set_xlim(0, 2050)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel('Iteration', labelpad=4)
    ax.grid(color='#E3E7EC', lw=.7)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')


def main():
    # Only this independent figure directory is written, even on layout-only reruns.
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['panels', 'data', 'code']:
        (OUT / name).mkdir(exist_ok=True)
    sources = json.loads((SOURCE / 'sources.json').read_text(encoding='utf-8'))
    guard = json.loads((SOURCE / 'guard_status.json').read_text(encoding='utf-8'))
    visibility, environment = rows('visibility_metrics.csv'), rows('environment_metrics.csv')
    opacity = rows('occlusion_gaussians.csv')
    count = len(opacity)
    changed = sum(abs(float(r['alpha_initial']) - float(r['alpha_full_candidate'])) > 1e-6 for r in opacity)
    reduced = sum(float(r['alpha_initial']) - float(r['alpha_full_candidate']) > 1e-6 for r in opacity)
    assert count == sources['occlusion_count'] == 313
    assert changed == reduced == sources['opacity_statistics']['C']['n_reduced_over_1e6']
    assert all(not v['candidate_accepted'] and v['guarded_selected_step'] == 0 for v in guard.values())
    projection = np.load(SOURCE / 'occlusion_projection.npz')

    image_names = ['A_scene.png', 'B_scene.png', 'C_scene.png', 'A_base.png', 'B_base.png', 'C_base.png',
                   'old_mask.png', 'new_mask.png', 'environment_mask.png', 'occluders_projection.png']
    data_names = ['visibility_metrics.csv', 'environment_metrics.csv', 'anchor_base_visibility.csv',
                  'occlusion_gaussians.csv', 'occlusion_projection.npz', 'guard_status.json']
    used = [SOURCE / 'panels' / n for n in image_names] + [SOURCE / n for n in data_names]
    used += [SOURCE / 'sources.json', SOURCE / 'figure_F14_revised.png', SOURCE / 'figure_F14_revised.pdf']
    source_hashes = {str(p): sha(p) for p in used}
    for name in image_names:
        shutil.copy2(SOURCE / 'panels' / name, OUT / 'panels' / name)
    for name in data_names:
        shutil.copy2(SOURCE / name, OUT / 'data' / name)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 13.5, 'axes.titlesize': 14,
        'axes.labelsize': 13.5, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
        'legend.fontsize': 13.5, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    width, height = 14.8, 12.45
    fig = plt.figure(figsize=(width, height), facecolor='white')

    def ax(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    def label(x, y, text, **kwargs):
        return fig.text(x / width, y / height, text, **kwargs)

    main = np.asarray(sources['main_crop_xyxy_at1024'])
    detail = np.asarray(sources['detail_crop_xyxy_at1024'])
    metric = {r['method']: r['base_visibility'] for r in sources['local_visibility']}
    inset_records = []
    for key, x, title, step in zip('ABC', [.25, 5.10, 9.95],
                                  ['(a) Initial Integration', '(b) RGB-only Refinement', '(c) Full M4 Candidate'], [0, 2000, 2000]):
        scene = ax(x, 6.94, 4.6, 4.6)
        pixels = image(scene, key + '_scene.png')
        scale = pixels.shape[1] / (main[2] - main[0])
        xy, wh = (detail[:2] - main[:2]) * scale, (detail[2:] - detail[:2]) * scale
        scene.add_patch(Rectangle(xy, *wh, fill=False, edgecolor='white', lw=1.25, linestyle=(0, (4, 3))))
        inset_width = 4.6 * .49
        inset_height = inset_width * 659 / 1000
        inset_box = [x + 4.6 - inset_width, 6.94 + 4.6 - inset_height,
                     inset_width, inset_height]
        zoom = ax(*inset_box)
        image(zoom, key + '_base.png')
        # Keep the entire 1.6 pt stroke inside the inset, with its outer edge
        # exactly flush to the scene's top and right boundaries.
        half_stroke_inches = 1.6 / 72 / 2
        px, py = half_stroke_inches / inset_width, half_stroke_inches / inset_height
        zoom.add_patch(Rectangle((px, py), 1 - 2 * px, 1 - 2 * py, transform=zoom.transAxes, fill=False,
                                edgecolor='#E53935', lw=1.6, clip_on=False))
        label(x + 2.3, 12.12, title, ha='center', weight='bold', fontsize=16)
        label(x + 2.3, 11.85, f'Step {step}', ha='center', fontsize=13)
        label(x + 2.3, 6.68, f'Base Visibility: {metric[key] * 100:.1f}%', ha='center', fontsize=13.5)
        inset_records.append({'method': key, 'bounds_inches': inset_box, 'crop_xyxy_at1024': detail.tolist()})

    label(.25, 6.22, '(d) Dual-Mask Supervision', fontsize=16, weight='bold')
    label(7.68, 6.22, '(e) Occluding Gaussian Adjustment', fontsize=16, weight='bold')
    mechanism_y, mechanism_h = 3.52, 2.17
    for x, name, title in zip([.25, 2.58, 4.91], ['old_mask.png', 'new_mask.png', 'environment_mask.png'],
                              ['Old Visible Mask', 'New Object Mask', 'Environment Mask']):
        a = ax(x, mechanism_y, mechanism_h, mechanism_h)
        image(a, name)
        a.set_title(title, fontsize=13, pad=9)
    label(3.665, 3.22, r'$M_{\mathrm{env}}=1-(M_{\mathrm{old}}\cup M_{\mathrm{new}})$',
          ha='center', fontsize=15)

    mechanism_w = mechanism_h * 1000 / 659
    a = ax(7.68, mechanism_y, mechanism_w, mechanism_h)
    image(a, 'occluders_projection.png')
    a.set_title('Step 0 Occlusion\nContribution', fontsize=13, pad=8)
    a = ax(11.225, mechanism_y, mechanism_w, mechanism_h)
    # This is an absolute, contribution-weighted alpha difference, NOT min-max normalization.
    visible = projection['initial_candidate_contribution'] > .003
    data = np.ma.masked_where(~visible, projection['mean_opacity_reduction'])
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#F1F2F4')
    im = a.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    a.set_axis_off()
    a.set_title('Background Gaussian\nOpacity Reduction', fontsize=13, pad=8)
    cb = fig.colorbar(im, cax=ax(11.55, 3.29, 2.64, .085), orientation='horizontal', ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=11, length=2.5, pad=2)
    label(12.87, 2.92, r'Absolute decrease: $\langle\alpha_0-\alpha_{2000}\rangle_w$', ha='center', fontsize=11.5)
    label(10.97, 2.66, f'Candidates: {count}  |  Opacity reduced: {reduced}', ha='center', fontsize=12.5)

    label(.92, 2.35, '(f) Trusted Object Visibility', fontsize=16, weight='bold')
    label(8.16, 2.35, '(g) Environment Preservation', fontsize=16, weight='bold')
    a = ax(.92, .65, 5.99, 1.48)
    curve(a, visibility, 'trusted_visibility', 100)
    a.set_ylabel('Trusted Object\nVisibility (%)', labelpad=6)
    a.yaxis.set_major_locator(MultipleLocator(1))
    a = ax(8.16, .65, 5.99, 1.48)
    curve(a, environment, 'heldout_environment_rgb_l1', 1)
    a.set_ylabel('Held-out RGB L1', labelpad=7)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    a.yaxis.set_major_locator(MultipleLocator(.0002))
    fig.legend(handles=[Line2D([], [], color=color, lw=2.15, label=method) for method, color in COLORS.items()],
               loc='lower center', bbox_to_anchor=(.5, -.002), ncol=2, frameon=False, columnspacing=2.6)

    fig.savefig(OUT / 'figure_F14_final.png', dpi=300)
    fig.savefig(OUT / 'figure_F14_final.pdf')
    for name, bounds in {
        'a_initial_integration': (.12, 6.48, 4.85, 5.80),
        'b_rgb_only': (4.97, 6.48, 4.85, 5.80),
        'c_full_m4_candidate': (9.82, 6.48, 4.85, 5.80),
        'd_dual_mask_supervision': (.12, 3.03, 7.10, 3.35),
        'e_occluding_gaussian_adjustment': (7.54, 2.51, 7.12, 3.87),
        'f_trusted_object_visibility': (.05, .14, 7.18, 2.38),
        'g_environment_preservation': (7.29, .14, 7.18, 2.38),
    }.items():
        fig.savefig(OUT / 'panels' / (name + '.png'), dpi=300, bbox_inches=Bbox.from_bounds(*bounds))
    plt.close(fig)

    caption = (
        'Fig. 14. Residual Gaussian occlusion refinement for the fixed Case00 industrial robot. '
        '(a-c) Same-camera scene renders before optimization, after the RGB-only control, '
        'and after full M4 candidate optimization. Each upper-right inset magnifies the identical base region '
        'marked by the white dashed rectangle. Base visibility is 52.2%, 52.9%, and 93.6%, respectively. '
        '(d) Cached old-visible, new-object, and complementary environment masks. '
        '(e) Actual initial occluder contributions and the contribution-weighted absolute opacity decrease, without '
        'min-max normalization. All 313 candidate background Gaussians have reduced opacity (change greater than 1e-6); '
        'none are deleted, and the new device remains fixed. '
        '(f-g) Unsmoothed trusted visibility and environment RGB L1 evaluations over the same ten held-out photographic '
        'views, recorded every 50 steps. The cached anchor masks illustrate the partition; real-photo RGB supervision '
        'uses registered photographic views. The existing RGB-only control disables visibility, occlusion-specific '
        'environment/foreground terms, and occlusion regularization while retaining environment RGB and parameter '
        'regularization. The full candidate improves visibility, but both optimized candidates fail the unchanged '
        'per-view environment-preservation guards. Neither is accepted: Guarded Final retains Step 0. '
        'The mean environment curve does not replace these per-view checks. All panels reuse existing renders and '
        'experiment records; no training, Gaussian editing, or new experiment was performed.\n'
    )
    (OUT / 'caption.txt').write_text(caption, encoding='utf-8')
    sources['previous_figure_layout'] = sources.pop('figure', None)
    sources['figure'] = {
        'script': str(Path(__file__)), 'source_directory': str(SOURCE),
        'canvas_inches': [width, height], 'png_dpi': 300,
        'layout': 'Three same-camera full-device scenes with upper-right base-detail insets; unchanged masks, attribution, and curves.',
        'no_rerendering': True, 'no_new_training': True, 'unchanged_experiment_values': True,
        'annotations': 'White dashed base-region rectangle on each scene; red inset borders flush to the top and right scene edges; no arrows.',
        'insets': inset_records,
        'image_crops': 'Unchanged from the preceding F14; identical detail crop for all three methods and both mechanism maps.',
        'opacity_map_normalized': False, 'opacity_map_color_range': [0, 1],
        'opacity_map_display_contribution_threshold': .003,
        'candidate_count': count, 'opacity_changed_count': changed, 'opacity_reduced_count': reduced,
        'opacity_change_count_tolerance': 1e-6, 'curves_unsmoothed': True,
        'candidate_not_guarded_final': True, 'no_guard_threshold_changes': True,
        'input_files_sha256': source_hashes,
    }
    sources['figure']['input_files_preserved'] = all(sha(Path(p)) == h for p, h in source_hashes.items())
    assert sources['figure']['input_files_preserved']
    sources['figure']['script_sha256'] = sha(Path(__file__))
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    (OUT / 'README.txt').write_text(
        'F14 layout-only final: existing Case00 results, no new rendering or optimization.\n'
        'panels/: unchanged source rasters and seven numbered composition panels.\n'
        'data/: exact copied evaluations, opacity table/map, and guard status.\n'
        'sources.json: original cameras/crops and PLY provenance plus the current layout metadata.\n'
        'Opacity map: absolute contribution-weighted decrease, not normalized. Count tolerance: 1e-6.\n'
        'Full M4 Candidate is rejected by the unchanged guard; Guarded Final remains Step 0.\n'
        'Reproduce: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f14_occlusion_insets.py\n',
        encoding='utf-8')
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2400', '-png',
                    str(OUT / 'figure_F14_final.pdf'), str(OUT / 'pdf_preview')], check=True)
    print('F14 FINAL RELAYOUT COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
