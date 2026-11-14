"""Case00 spatial-mechanism figure from immutable cached experiment outputs.

CPU plotting only: reuse the continuous, scene-occluded M1 Gaussian contribution.
The approved historical Qwen image is not relabelled as an adaptive-condition run.
"""
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
PREVIOUS = BASE/'F11_spatial_conditioning_case00'
REVISED = BASE/'F11_spatial_conditioning_revised'
SOURCE = PREVIOUS/'inputs'
OUT = BASE/'F11_spatial_conditioning_final'
ORANGE, PURPLE, CYAN, GREEN = '#E9A32F', '#D42BCE', '#00BEC9', '#70B494'
WORKSPACE_CROP = [90, 180, 934, 1024]
CONDITION_CROP = [0, 340, 1024, 900]
DETAIL_CROP = [270, 710, 670, 880]
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


def corners(low, high):
    return np.asarray(list(itertools.product(*zip(low, high))))


def image(path, data):
    Image.fromarray(np.rint(np.clip(data, 0, 255)).astype('uint8')).save(path)


def main():
    for sub in ['inputs', 'analysis', 'panels', 'code']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    protected = {str(p): sha(p) for folder in [PREVIOUS, REVISED]
                 for p in folder.rglob('*') if p.is_file()}
    previous_sources = read(PREVIOUS/'sources.json')
    conditions = read(SOURCE/'conditions.json')
    region = read(SOURCE/'generation_region.json')
    camera = conditions['camera']
    assert camera == read(SOURCE/'anchor_camera.json')
    assert camera == read(SOURCE/'confirmed_generation_conditions.json')['camera']
    assert sha(SOURCE/'generation_region.json') == conditions['generation_region_sha256']
    for p in SOURCE.iterdir():
        if p.is_file():
            assert sha(p) == previous_sources['source_files'][p.name]['sha256']
            shutil.copy2(p, OUT/'inputs'/p.name)

    generation_source = read(ROOT/'experiment figure/F05/sources.json')['qwen']
    assert sha(SOURCE/'edited_scene.png') == generation_source['sha256']
    assert sha(SOURCE/'edit_mask.png') != sha(SOURCE/'confirmed_generation_edit_mask.png')
    m1 = Path(previous_sources['m1_directory'])
    for name in ['target_indices.npy', 'target_gaussians.ply']:
        assert sha(m1/name) == conditions['input_sha256'][name]
    selected_count = len(np.load(m1/'target_indices.npy'))
    assert selected_count == read(SOURCE/'m1_target.json')['target_gaussians']

    # This map was produced by rasterizing binary per-Gaussian colors in the
    # complete scene: continuous transmittance-weighted selected contribution,
    # preserving real occlusion. No binary silhouette fill or fake splat texture.
    map_source = Path(previous_sources['m2_directory'])/'spatial_depth.npz'
    with np.load(map_source) as data:
        contribution = data['target_visible_alpha'].copy()
    old_visible = np.asarray(Image.open(SOURCE/'old_visible_mask.png')) > 0
    assert np.array_equal(contribution > region['settings']['old_alpha_threshold'], old_visible)
    assert np.isfinite(contribution).all() and contribution.min() >= 0 and contribution.max() <= 1.0001
    np.save(OUT/'inputs/target_visible_alpha.npy', contribution)
    Image.fromarray(np.rint(np.clip(contribution, 0, 1)*65535).astype('uint16')).save(
        OUT/'analysis/target_visible_alpha_16bit.png')

    rgb = np.asarray(Image.open(SOURCE/'anchor.png').convert('RGB'))
    generated = np.asarray(Image.open(SOURCE/'edited_scene.png').convert('RGB'))
    hint = np.asarray(Image.open(SOURCE/'position_hint.png').convert('RGB'))
    depth = np.asarray(Image.open(SOURCE/'depth_background.png').convert('L'))
    edit = np.asarray(Image.open(SOURCE/'edit_mask.png')) > 0
    guard = np.asarray(Image.open(SOURCE/'protected_foreground_mask.png')) > 0
    assert rgb.shape == generated.shape == hint.shape == (1024, 1024, 3)
    assert not (edit & guard).any() and not depth[edit].any()
    yy, xx = np.where(edit)
    assert (xx.min() >= CONDITION_CROP[0] and xx.max() < CONDITION_CROP[2]
            and yy.min() >= CONDITION_CROP[1] and yy.max() < CONDITION_CROP[3])
    assert (CONDITION_CROP[0] <= DETAIL_CROP[0] < DETAIL_CROP[2] <= CONDITION_CROP[2]
            and CONDITION_CROP[1] <= DETAIL_CROP[1] < DETAIL_CROP[3] <= CONDITION_CROP[3])
    color = np.asarray(matplotlib.colors.to_rgb(ORANGE))*255
    alpha = .52*np.clip(contribution, 0, 1)[..., None]
    localized = rgb*(1-alpha) + color*alpha
    image(OUT/'analysis/m1_gaussian_highlight_full.png', localized)
    # Fade the SAVED position hint towards its original RGB only for display.
    # Its 20% fill becomes 12%; both the input image and all masks stay intact.
    condition_display = rgb.astype(float) + .60*(hint.astype(float)-rgb.astype(float))
    condition_display[guard] = .76*rgb[guard] + .24*np.asarray(matplotlib.colors.to_rgb(CYAN))*255
    image(OUT/'analysis/condition_display_full.png', condition_display)

    # Reuse exactly the accepted depth contrast and absolute difference arrays.
    depth_display = np.asarray(Image.open(REVISED/'analysis/depth_display_contrast.png'))
    assert not depth_display[edit].any()
    difference = np.abs(generated.astype(np.float32)-rgb.astype(np.float32)).mean(2)/255.
    metrics = read(REVISED/'analysis/metrics.json')
    assert abs(float(difference[edit].mean())-metrics['inside_edit_rgb_mae']) < 1e-9
    assert abs(float(difference[~edit].mean())-metrics['outside_edit_rgb_mae']) < 1e-9
    assert np.array_equal(difference, np.load(REVISED/'analysis/rgb_difference.npy'))
    np.save(OUT/'analysis/rgb_difference.npy', difference)
    write(OUT/'analysis/metrics.json', metrics)
    shutil.copy2(REVISED/'analysis/depth_display_contrast.png', OUT/'analysis/depth_display_contrast.png')

    geometry = region['geometry']
    old_local = corners(geometry['old_local_low'], geometry['old_local_high'])
    new_local = corners(geometry['new_local_low'], geometry['new_local_high'])
    axes, origin = np.asarray(geometry['axes']), np.asarray(geometry['origin'])
    new_world = new_local @ axes.T + origin
    assert np.max(np.linalg.norm(new_world-np.asarray(region['bbox_corners']), axis=1)) < 1e-7
    new_xy = project(new_world, camera)
    support_ids = [0, 1, 5, 4]  # local y is up; this is the actual lower face.
    support_xy = new_xy[support_ids]
    box_mask = np.zeros((1024, 1024), np.uint8)
    cv2.fillConvexPoly(box_mask, cv2.convexHull(np.round(new_xy).astype('int32')), 1)
    assert np.array_equal(box_mask > 0, np.asarray(Image.open(SOURCE/'generation_bbox_mask.png')) > 0)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'pdf.fonttype': 42})
    W, H = 180., 146.
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')

    def axis(x, y, w, h, **kw):
        return fig.add_axes([x/W, y/H, w/W, h/H], **kw)

    def text(x, y, value, **kw):
        return fig.text(x/W, y/H, value, **kw)

    def display(ax, data, crop, **kw):
        if data.ndim == 3:
            data = np.rint(np.clip(data, 0, 255)).astype('uint8')
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
    display(a, localized, WORKSPACE_CROP)
    a.add_patch(Polygon(support_xy, closed=True, facecolor=GREEN, edgecolor=GREEN, alpha=.25, linewidth=.7))
    wire(a, new_xy, PURPLE, 1.0)
    # A separate ORTHOGRAPHIC schematic uses a single unchanged M2 local frame.
    # Display coordinates are (local x, local z, local y) to draw y vertically.
    schema = axis(3.2, 113.3, 23.5, 23.5, projection='3d')
    schema.set_facecolor('white')
    schema.set_proj_type('ortho')
    for points, color_, ls in [(old_local, ORANGE, (0, (2.6, 1.6))), (new_local, PURPLE, 'solid')]:
        q = points[:, [0, 2, 1]]
        for i, j in EDGES:
            schema.plot(*q[[i, j]].T, color=color_, lw=.68, linestyle=ls)
    schema.add_collection3d(Poly3DCollection([new_local[support_ids][:, [0, 2, 1]]],
                            facecolors=GREEN, edgecolors=GREEN, alpha=.32, linewidths=.4))
    combined = np.concatenate([old_local, new_local])[:, [0, 2, 1]]
    lower, upper = combined.min(0), combined.max(0)
    span = upper-lower
    for setter, low, high, dim in zip([schema.set_xlim, schema.set_ylim, schema.set_zlim], lower, upper, span):
        setter(low-.04*dim, high+.04*dim)
    schema.set_box_aspect(span)
    schema.view_init(elev=20, azim=-58)
    schema.set_axis_off()
    text(14.95, 135.9, 'M2 local frame', fontsize=5.1, ha='center', va='center')
    for yy_, handle, label in [
        (65.4, Patch(facecolor=ORANGE, alpha=.55), 'M1 Localized Equipment'),
        (61.9, Line2D([0], [0], color=PURPLE, lw=1.1), 'M2 Generation Space'),
        (58.4, Patch(facecolor=GREEN, alpha=.55), 'Support Plane')]:
        fig.legend(handles=[handle], labels=[label], bbox_to_anchor=(2/W, yy_/H), loc='center left',
                   frameon=False, fontsize=6.2, borderaxespad=0, handlelength=1.6, handletextpad=.6)

    main_h = 103.*(CONDITION_CROP[3]-CONDITION_CROP[1])/(CONDITION_CROP[2]-CONDITION_CROP[0])
    b = axis(75, 138-main_h, 103, main_h)
    display(b, condition_display, CONDITION_CROP)
    b.contour(guard, levels=[.5], colors=[CYAN], linewidths=.55)
    x0, y0, x1, y1 = DETAIL_CROP
    b.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='white',
                          linestyle=(0, (3, 2)), linewidth=.85))
    bw = 50.25
    bh = bw*(y1-y0)/(x1-x0)
    detail_axes = []
    for x, label, data in [(75, 'Editing Condition', condition_display),
                           (127.75, 'Generated Reference*', generated)]:
        text(x+bw/2, 79.6, label, fontsize=6.6, va='center', ha='center')
        bx = axis(x, 56, bw, bh)
        display(bx, data, DETAIL_CROP)
        if label == 'Editing Condition':
            bx.contour(guard, levels=[.5], colors=[CYAN], linewidths=.8)
        for sp in bx.spines.values():
            sp.set_visible(True); sp.set_color('#c9c9c9'); sp.set_linewidth(.45)
        detail_axes.append(bx)
    fig.legend(handles=[Line2D([0], [0], color=PURPLE, lw=3, label='Editable Region'),
                        Line2D([0], [0], color=CYAN, lw=2, label='Protected Foreground')],
               bbox_to_anchor=(126.5/W, 52.4/H), loc='center', ncol=2, frameon=False,
               fontsize=6.2, handlelength=1.6, handletextpad=.5, columnspacing=1.5)

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
    cbax = axis(139, 4.8, 36, 1.15)
    colorbar = fig.colorbar(panels_d[2].images[0], cax=cbax, orientation='horizontal', ticks=[0, .25, .5])
    colorbar.ax.set_xticklabels(['0', '0.25', '0.5+'])
    colorbar.ax.tick_params(labelsize=5.1, length=1.3, pad=1.)
    colorbar.outline.set_visible(False)

    fig.savefig(OUT/'figure_F11_final.png', dpi=300)
    fig.savefig(OUT/'figure_F11_final.pdf')
    panel_boxes = {
        'a_3d_workspace_construction': [1, 55.8, 72.5, 145],
        'b_visibility_aware_editing': [74, 50, 179, 145],
        'c_background_depth': [1, 2.7, 45, 48.6],
        'd_localized_editing_analysis': [46, .2, 179, 48.6],
        'b_editing_condition': [74.6, 55.6, 125.65, 81.6],
        'b_generated_reference': [127.35, 55.6, 178.4, 81.6],
        'a_m2_local_frame_schematic': [3.2, 113.3, 26.7, 137.8],
    }
    for name, box in panel_boxes.items():
        fig.savefig(OUT/'panels'/(name+'.png'), dpi=600,
                    bbox_inches=Bbox.from_extents(*[v/25.4 for v in box]))
    for name, arr, crop in [('d_original', rgb, ANALYSIS_CROP), ('d_generated', generated, ANALYSIS_CROP),
                           ('b_generated_crop_unannotated', generated, DETAIL_CROP)]:
        Image.fromarray(arr).crop(crop).save(OUT/'panels'/(name+'.png'), dpi=(300, 300))
    heat = matplotlib.colormaps['inferno'](np.clip(difference/.5, 0, 1))[:, :, :3]*255
    image(OUT/'analysis/difference_heatmap_full.png', heat)
    Image.open(OUT/'analysis/difference_heatmap_full.png').crop(ANALYSIS_CROP).save(OUT/'panels/d_rgb_difference.png')
    plt.close(fig)

    caption = (
        'Fig. 11. Spatial workspace construction and visibility-aware editing conditions for Case00. '
        '(a) Orange denotes the cached continuous visible contribution of the actual M1-selected Gaussians '
        '(not a filled segmentation mask), overlaid on the original scene. Magenta outlines the M2 generation volume '
        'and translucent green marks its lower support face. The inset compares the estimated original extent '
        '(orange dashed) and replacement generation space (magenta solid) in the same M2 local coordinate frame; '
        'their dimensions are not taken from different coordinate systems. '
        '(b) Saved position-hint conditions with reduced display opacity and cyan protected-foreground contours. '
        'The dashed box identifies the identical pixel crop shown below as Editing Condition and Generated Reference. '
        'The latter is unmodified and has no mask overlay. '
        '(c) Background depth with a display-only contrast adjustment; the black replacement region has unknown depth. '
        '(d) Identically cropped original and generated RGB and their mean per-channel absolute difference in [0,1] '
        '(heatmap clipped at 0.5). Full-image inside-edit and outside-edit differences remain 0.216 and 0.018. '
        '*Both generated views use the same previously approved main-pipeline Qwen image, produced with earlier '
        'projected-box conditions. It is not a direct output of the later adaptive conditions visualized here. '
        'All use the same saved Anchor camera and native 1024 x 1024 canvas. The foreground comparison illustrates '
        'the recorded conditions and historical output, without attributing that output to the newer protection mask. '
        'No localization, generation, training, or new Gaussian rendering was performed.')
    (OUT/'caption.txt').write_text(caption+'\n', encoding='utf-8')
    manifest = dict(
        case='case00', previous_sources=previous_sources, prior_revised_figure=str(REVISED),
        source_hashes={p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()},
        camera=camera, native_resolution=[1024, 1024],
        gaussian_overlay=dict(
            source_npz=str(map_source), source_npz_sha256=sha(map_source), key='target_visible_alpha',
            stored_array=str(OUT/'inputs/target_visible_alpha.npy'),
            array_sha256=sha(OUT/'inputs/target_visible_alpha.npy'),
            original_rasterizer='industrial_swap_v2/core/m2_edit_mask.py:render_geometry',
            original_definition='Full-scene rasterization of selected-Gaussian flags; actual alpha and background occlusion',
            agrees_exactly_with_old_visible_mask_at_saved_threshold=True,
            selected_count=selected_count, selected_indices=str(m1/'target_indices.npy'),
            selected_ply=str(m1/'target_gaussians.ply'),
            selected_indices_sha256=conditions['input_sha256']['target_indices.npy'],
            selected_ply_sha256=conditions['input_sha256']['target_gaussians.ply'],
            color=ORANGE, display_gain=.52, no_binary_fill=True,
            parameters_changed=False, new_render=False),
        geometry=dict(
            source='inputs/generation_region.json',
            main_scene_old_bbox_drawn=False,
            schematic_old_source='geometry.old_local_low/high',
            schematic_new_source='geometry.new_local_low/high',
            schematic_uses_identical_local_frame=True, schematic_coordinate_order=['x', 'z', 'y'],
            schematic_projection='orthographic, elevation 20 degrees, azimuth -58 degrees',
            schematic_equal_metric_scaling=True, old_local_corners=old_local.tolist(),
            new_local_corners=new_local.tolist(), world_axes=axes.tolist(), world_origin=origin.tolist(),
            new_world_corners=new_world.tolist(), new_projected_corners=new_xy.tolist(),
            support_corner_indices=support_ids, support_is_new_local_y_minimum=True,
            m1_bbox_not_used_as_schematic_extent=True),
        visualization=dict(
            workspace_crop=WORKSPACE_CROP, complete_edit_region_crop=CONDITION_CROP,
            identical_detail_crops=DETAIL_CROP, depth_crop=DEPTH_CROP, identical_analysis_crops=ANALYSIS_CROP,
            position_hint_display='RGB + 0.60*(saved position_hint - RGB); actual masks unmodified',
            editable_fill_alpha=.12, protected_foreground='Unmodified full protected_foreground_mask.png',
            protected_pixels=int(guard.sum()), protection_edit_overlap=int((guard & edit).sum()),
            depth_contrast=read(REVISED/'sources.json')['depth_contrast'],
            figure_mm=[W, H], dpi=300, panel_dpi=600,
            top_column_widths_mm=[69.5, 103], bottom_widths_mm=[42, 131]),
        generated_source=generation_source, same_generated_image_in_b_and_d=True,
        generated_from_current_adaptive_conditions=False,
        generation_provenance='Approved historical projected-box output; later adaptive conditions are shown separately',
        statistics=metrics, no_model_or_algorithm_runs=True,
        qwen_calls=0, trellis_calls=0, m1_m4_runs=0, new_gaussian_renders=0)
    write(OUT/'sources.json', manifest)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    write(OUT/'validation.json', dict(
        old_figures_unchanged=True, experiment_inputs_unchanged=True,
        real_continuous_selected_gaussian_contribution=True,
        selected_ply_and_indices_match_prepared_condition_hashes=True,
        same_m2_coordinate_system_for_old_new_schematic=True,
        main_no_old_bbox=True, generation_box_matches_saved_mask=True,
        full_edit_region_visible=True, protected_mask_not_modified=True,
        b_and_d_same_approved_generation=True, camera_and_native_rgb_resolution_identical=True,
        difference_map_and_metrics_unchanged=True, original_qwen_depth_unchanged=True,
        historical_generation_provenance_disclosed=True))
    shutil.copy2(__file__, OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2200', '-png',
                    str(OUT/'figure_F11_final.pdf'), str(OUT/'pdf_proof')], check=True)
    (OUT/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>F11 spatial conditioning - Case00</title>'
        '<style>body{font:16px/1.5 Arial;max-width:1500px;margin:24px auto}img{max-width:100%}</style>'
        '<p>Case00 - real cached Gaussian contribution, local-frame workspace schematic and foreground detail.</p>'
        '<p><a href="figure_F11_final.png">PNG</a> | <a href="figure_F11_final.pdf">PDF</a> | '
        '<a href="panels/">Panels</a> | <a href="caption.txt">Caption</a> | <a href="sources.json">Sources</a></p>'
        '<img src="figure_F11_final.png">', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), gaussians=selected_count,
                          reused_continuous_contribution=True, preserved_old_versions=True)), flush=True)


if __name__ == '__main__':
    main()
