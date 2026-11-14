"""F11 presentation revision: original M1 contour/box and original F details.

No experiment or renderer runs. Both boxes are projected in the same saved
Anchor camera after their original data have been expressed in world space.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
OLD = BASE/'F11_spatial_conditioning_case00'
PREVIOUS = BASE/'F11_spatial_conditioning_final'
SOURCE = PREVIOUS/'inputs'
OUT = BASE/'F11_spatial_conditioning_final_v2'
ORANGE, PURPLE, CYAN, GREEN = '#EFAD36', '#D42BCE', '#00BEC9', '#70B494'
WORKSPACE_CROP = [90, 180, 934, 1024]
CONDITION_CROP = [0, 270, 1024, 910]
DETAIL_CROP = [300, 690, 500, 890]
DEPTH_CROP = [100, 270, 940, 900]
ANALYSIS_CROP = [170, 240, 930, 810]
EDGES = [(i, i ^ b) for i in range(8) for b in (1, 2, 4) if i < (i ^ b)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def project(points, camera):
    q = np.asarray(points) @ np.asarray(camera['R']).T + camera['t']
    assert np.all(q[:, 2] > 0)
    return q[:, :2]/q[:, 2:] * [camera['fx'], camera['fy']] + [camera['cx'], camera['cy']]


def main():
    for sub in ['inputs', 'analysis', 'panels', 'code']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    protected = {str(p): sha(p) for folder in [OLD, PREVIOUS]
                 for p in folder.rglob('*') if p.is_file()}
    for p in SOURCE.iterdir():
        if p.is_file():
            shutil.copy2(p, OUT/'inputs'/p.name)
    previous_sources, old_sources = read(PREVIOUS/'sources.json'), read(OLD/'sources.json')
    camera = read(SOURCE/'anchor_camera.json')
    assert camera == previous_sources['camera'] == old_sources['camera']
    rgb = np.asarray(Image.open(SOURCE/'anchor.png').convert('RGB'))
    generated = np.asarray(Image.open(SOURCE/'edited_scene.png').convert('RGB'))
    visible = np.asarray(Image.open(SOURCE/'old_visible_mask.png')) > 0
    # Use the original F panel's prepared display and original local guard.
    hint = np.asarray(Image.open(OLD/'analysis/position_hint_with_protection.png').convert('RGB'))
    guard = np.asarray(Image.open(OLD/'analysis/protected_local_mask.png')) > 0
    expected_guard = ((np.asarray(Image.open(SOURCE/'protected_foreground_mask.png')) > 0)
                      & (np.asarray(Image.open(SOURCE/'extension_proposal.png')) > 0))
    assert np.array_equal(guard, expected_guard)
    assert np.array_equal(np.asarray(Image.fromarray(hint).crop(DETAIL_CROP)),
                          np.asarray(Image.open(OLD/'analysis/f_hint.png')))
    assert np.array_equal(np.asarray(Image.fromarray(generated).crop(DETAIL_CROP)),
                          np.asarray(Image.open(OLD/'analysis/f_generated.png')))
    for name in ['position_hint_with_protection.png', 'protected_local_mask.png', 'f_hint.png', 'f_generated.png']:
        shutil.copy2(OLD/'analysis'/name, OUT/'analysis'/name)
    for name in ['depth_display_contrast.png', 'rgb_difference.npy', 'metrics.json']:
        shutil.copy2(PREVIOUS/'analysis'/name, OUT/'analysis'/name)
    depth_display = np.asarray(Image.open(OUT/'analysis/depth_display_contrast.png'))
    difference = np.load(OUT/'analysis/rgb_difference.npy')
    metrics = read(OUT/'analysis/metrics.json')
    assert np.array_equal(difference, np.abs(generated.astype(np.float32)-rgb.astype(np.float32)).mean(2)/255.)
    assert sha(SOURCE/'edited_scene.png') == previous_sources['generated_source']['sha256']

    region = read(SOURCE/'generation_region.json')
    m1 = read(SOURCE/'m1_target.json')
    # Explicitly restore the M1 localization box, not M2's measured old extent.
    old_world = np.asarray(m1['bbox_corners'])
    new_world = np.asarray(region['bbox_corners'])
    old_xy, new_xy = project(old_world, camera), project(new_world, camera)
    assert np.allclose(old_xy, old_sources['projected_geometry']['m1_bbox_corners_pixels'], atol=1e-7)
    support_xy = new_xy[[0, 1, 5, 4]]
    for points in [old_xy, new_xy]:
        assert (points[:, 0].min() > WORKSPACE_CROP[0] and points[:, 0].max() < WORKSPACE_CROP[2]
                and points[:, 1].min() > WORKSPACE_CROP[1] and points[:, 1].max() < WORKSPACE_CROP[3])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'pdf.fonttype': 42})
    W, H = 180., 146.
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')

    def axis(x, y, w, h):
        return fig.add_axes([x/W, y/H, w/W, h/H])

    def text(x, y, value, **kw):
        return fig.text(x/W, y/H, value, **kw)

    def display(ax, data, crop, **kw):
        ax.imshow(data, extent=[-.5, 1023.5, 1023.5, -.5], interpolation='lanczos', **kw)
        ax.set_xlim(crop[0]-.5, crop[2]-.5)
        ax.set_ylim(crop[3]-.5, crop[1]-.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def wire(ax, pts, color, lw, style='solid'):
        for i, j in EDGES:
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], color=color, lw=lw,
                    linestyle=style, solid_capstyle='round', dash_capstyle='round')

    text(2, 142.3, '(a) 3D Workspace Construction', fontsize=7.5, fontweight='semibold', va='center')
    text(75, 142.3, '(b) Visibility-Aware Editing', fontsize=7.5, fontweight='semibold', va='center')
    a = axis(2, 68.5, 69.5, 69.5)
    display(a, rgb, WORKSPACE_CROP)
    a.add_patch(Polygon(support_xy, closed=True, facecolor=GREEN, edgecolor=GREEN, alpha=.25, linewidth=.7))
    a.contour(visible, levels=[.5], colors=[ORANGE], linewidths=.65)
    wire(a, old_xy, ORANGE, .9, (0, (3, 2)))
    wire(a, new_xy, PURPLE, 1.0)
    for yy, handle, label in [
        (65.4, Line2D([0], [0], color=ORANGE, lw=1.1), 'M1 Target Contour'),
        (61.9, Line2D([0], [0], color=ORANGE, lw=1.1, linestyle='--'), 'M1 Localization Box'),
        (58.4, Line2D([0], [0], color=PURPLE, lw=1.1), 'M2 Generation Space'),
        (54.9, Patch(facecolor=GREEN, alpha=.55), 'Support Plane')]:
        fig.legend(handles=[handle], labels=[label], bbox_to_anchor=(2/W, yy/H), loc='center left',
                   frameon=False, fontsize=6.2, borderaxespad=0, handlelength=1.6, handletextpad=.6)

    # A less panoramic main view leaves room for the original square F crops.
    bx, bw = 87.75, 77.5
    bh = bw*(CONDITION_CROP[3]-CONDITION_CROP[1])/(CONDITION_CROP[2]-CONDITION_CROP[0])
    b = axis(bx, 138-bh, bw, bh)
    display(b, hint, CONDITION_CROP)
    b.contour(guard, levels=[.5], colors=[CYAN], linewidths=.55)
    gx, gy, ex, ey = DETAIL_CROP
    b.add_patch(Rectangle((gx, gy), ex-gx, ey-gy, fill=False, edgecolor='white',
                          linestyle=(0, (3, 2)), linewidth=.7))
    for x, label, data in [(93.75, 'Position hint + guard', hint),
                           (128.25, 'Generated reference*', generated)]:
        text(x+15.5, 87.2, label, fontsize=5.9, va='center', ha='center')
        detail = axis(x, 54, 31, 31)
        display(detail, data, DETAIL_CROP)
        # Original F visualization: SAME saved guard on BOTH source and output.
        detail.contour(guard, levels=[.5], colors=[CYAN], linewidths=.6)
        for spine in detail.spines.values():
            spine.set_visible(True); spine.set_color('#c9c9c9'); spine.set_linewidth(.45)
    fig.legend(handles=[Line2D([0], [0], color=PURPLE, lw=3, label='Editable Region'),
                        Line2D([0], [0], color=CYAN, lw=2, label='Protected Foreground')],
               bbox_to_anchor=(126.5/W, 50.5/H), loc='center', ncol=2, frameon=False,
               fontsize=6.2, handlelength=1.6, handletextpad=.5, columnspacing=1.5)

    # Panels C/D keep their exact previous data, coordinates, crop and typography.
    text(2, 46.2, '(c) Background Depth', fontsize=7.4, fontweight='semibold', va='center')
    text(47, 46.2, '(d) Localized Editing Analysis', fontsize=7.4, fontweight='semibold', va='center')
    c = axis(2, 9, 42, 31.5)
    display(c, depth_display, DEPTH_CROP, cmap='gray', vmin=0, vmax=255)
    text(23, 5.0, 'Unknown Replacement Region', ha='center', va='center', fontsize=5.6)
    panels_d = []
    for x, label, data in zip([47, 91.5, 136], ['Original', 'Generated*', '|ΔRGB|'],
                              [rgb, generated, difference]):
        text(x+21, 42.5, label, ha='center', va='center', fontsize=6.6)
        da = axis(x, 9, 42, 31.5)
        display(da, data, ANALYSIS_CROP, **({'cmap': 'inferno', 'vmin': 0, 'vmax': .5} if data.ndim == 2 else {}))
        panels_d.append(da)
    text(47, 4.8, f"Inside-edit RGB Difference: {metrics['inside_edit_rgb_mae']:.3f}", fontsize=5.7, va='center')
    text(47, 1.9, f"Outside-edit RGB Difference: {metrics['outside_edit_rgb_mae']:.3f}", fontsize=5.7, va='center')
    colorbar = fig.colorbar(panels_d[2].images[0], cax=axis(139, 4.8, 36, 1.15),
                           orientation='horizontal', ticks=[0, .25, .5])
    colorbar.ax.set_xticklabels(['0', '0.25', '0.5+'])
    colorbar.ax.tick_params(labelsize=5.1, length=1.3, pad=1.)
    colorbar.outline.set_visible(False)

    fig.savefig(OUT/'figure_F11_final.png', dpi=300)
    fig.savefig(OUT/'figure_F11_final.pdf')
    for name, box in {
        'a_3d_workspace_construction': [1, 52.5, 72.5, 145],
        'b_visibility_aware_editing': [74, 48.7, 179, 145],
        'c_background_depth': [1, 2.7, 45, 48.6],
        'd_localized_editing_analysis': [46, .2, 179, 48.6],
        'b_position_hint_guard': [93.3, 53.6, 125.2, 89.1],
        'b_generated_reference': [127.8, 53.6, 159.7, 89.1],
    }.items():
        fig.savefig(OUT/'panels'/(name+'.png'), dpi=600,
                    bbox_inches=Bbox.from_extents(*[v/25.4 for v in box]))
    plt.close(fig)
    for name in ['d_original.png', 'd_generated.png', 'd_rgb_difference.png']:
        shutil.copy2(PREVIOUS/'panels'/name, OUT/'panels'/name)
    for name in ['c_background_depth.png', 'd_localized_editing_analysis.png']:
        assert np.array_equal(np.asarray(Image.open(OUT/'panels'/name)),
                              np.asarray(Image.open(PREVIOUS/'panels'/name))), name

    caption = (
        'Fig. 11. Spatial and visibility-aware conditioning for industrial equipment replacement (Case00). '
        '(a) The original M1 target contour (orange solid) and original M1 localization box (orange dashed) '
        'are shown with the M2 generation volume (magenta) and its lower support face (green). '
        'The M1 box uses m1_target.json bbox_corners; the M2 box uses generation_region.json bbox_corners. '
        'Both saved boxes are in scene world coordinates and projected using the same Anchor camera. '
        'There is no local-frame schematic and no substituted estimate of the M1 box. '
        '(b) The position hint and protected foreground with two square details restored from the original '
        'Foreground Preservation panel. The same saved cyan protection contour is overlaid on both detail '
        'images at identical pixel coordinates; it is not a new segmentation of the generated image. '
        '(c-d) Background depth and localized RGB analysis are unchanged from the preceding version. '
        'The unknown replacement region is black. Original, generated and absolute RGB-difference views '
        'share their saved crop; mean inside/outside differences are 0.216/0.018 on the full current edit-mask partition. '
        '*The confirmed robot reference was generated using earlier projected-box conditions, not the later '
        'adaptive conditions illustrated here. Both generated views use that same approved output and saved '
        'Anchor camera. No localization, generation, training, Gaussian geometry, or experimental masks were changed.')
    (OUT/'caption.txt').write_text(caption+'\n', encoding='utf-8')
    write(OUT/'sources.json', dict(
        case='case00', previous_figure_sources=previous_sources, original_seven_panel_sources=old_sources,
        camera=camera, native_resolution=[1024, 1024],
        source_hashes={p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()},
        changes=dict(a_no_inset=True, a_no_fill_overlay=True,
            a_contour_source='old_visible_mask.png, same level 0.5 and color as original panel a',
            a_m1_box_source='m1_target.json:bbox_corners (original world-axis-aligned localization box)',
            a_m2_box_source='generation_region.json:bbox_corners (world coordinates)',
            m1_world_corners=old_world.tolist(), m1_projected_corners=old_xy.tolist(),
            m2_world_corners=new_world.tolist(), m2_projected_corners=new_xy.tolist(),
            old_local_m2_extent_not_used=True, all_box_projections_use_same_anchor=True,
            b_restored_original_f_crop=DETAIL_CROP,
            b_detail_source_files={name: dict(path=str(OLD/'analysis'/name), sha256=sha(OLD/'analysis'/name))
                for name in ['f_hint.png', 'f_generated.png', 'protected_local_mask.png']},
            b_generated_has_same_saved_guard_contour=True,
            b_output_contour_is_annotation_not_generated_image_segmentation=True),
        crops=dict(workspace=WORKSPACE_CROP, condition=CONDITION_CROP, original_f=DETAIL_CROP,
                   depth=DEPTH_CROP, analysis=ANALYSIS_CROP),
        layout=dict(figure_mm=[W, H], dpi=300, panel_dpi=600,
                    b_main_size_mm=[bw,bh], b_main_aspect_ratio=1.6, b_detail_size_mm=[31,31]),
        c_d_pixel_identical_to_previous=True, statistics=metrics,
        generated_from_current_adaptive_conditions=False, same_approved_generated_image_in_b_and_d=True,
        qwen_calls=0, trellis_calls=0, training_runs=0, gaussian_renders=0))
    assert all(sha(Path(p)) == digest for p, digest in protected.items())
    write(OUT/'validation.json', dict(
        old_versions_unchanged=True, original_m1_box_restored=True,
        m1_box_projection_matches_original_e=True, original_target_contour_unchanged=True,
        a_no_schematic=True, f_crops_pixel_identical=True,
        original_guard_identical_in_both_details=True, c_d_panels_pixel_identical=True,
        experimental_inputs_unchanged=True, historical_generation_provenance_preserved=True))
    shutil.copy2(__file__, OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2200', '-png',
                    str(OUT/'figure_F11_final.pdf'), str(OUT/'pdf_proof')], check=True)
    (OUT/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>F11 revised contour and original F details</title>'
        '<style>body{font:16px/1.5 Arial;max-width:1500px;margin:24px auto}img{max-width:100%}</style>'
        '<p>Case00: original M1 contour and localization box; original Foreground Preservation crops.</p>'
        '<p><a href="figure_F11_final.png">PNG</a> | <a href="figure_F11_final.pdf">PDF</a> | '
        '<a href="panels/">Panels</a> | <a href="caption.txt">Caption</a> | <a href="sources.json">Sources</a></p>'
        '<img src="figure_F11_final.png">', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), c_d_unchanged=True, original_f_restored=True)), flush=True)


if __name__ == '__main__':
    main()
