"""F11 spatial-conditioning figure from immutable Case02 M1/M2 artifacts.

CPU-only: saved RGB, masks, depth and calibrated projection; no model invocation.
The candidate_inspection folder contains byte-preserved source copies and their
remote provenance. No M3/M4 image is used by this figure.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F11_spatial_conditioning'
INSPECTION = OUT / 'candidate_inspection'
CASE = 'case02'
SRC = INSPECTION / CASE
MAGENTA = '#D42BCE'
CYAN = '#00BEC9'
OLD = '#EFAD36'
SUPPORT = '#55A777'
EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]
GUARD_CROP = [748, 650, 1024, 926]
CHANGE_CROP = [285, 285, 945, 905]


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def project(points, camera):
    z = np.asarray(points) @ np.asarray(camera['R']).T + camera['t']
    assert (z[:, 2] > 0).all()
    return z[:, :2] / z[:, 2:] * [camera['fx'], camera['fy']] + [camera['cx'], camera['cy']]


def main():
    for sub in ['panels', 'inputs', 'analysis', 'code']:
        (OUT / sub).mkdir(exist_ok=True)
    manifest = read(INSPECTION / 'manifest.json')
    provenance = manifest[CASE]
    for name, record in provenance['source_files'].items():
        assert sha(SRC / name) == record['sha256'], name
        shutil.copy2(SRC / name, OUT / 'inputs' / name)
    condition = read(SRC / 'conditions.json')
    region = read(SRC / 'generation_region.json')
    m1 = read(SRC / 'm1_target.json')
    report = read(SRC / 'mask_report.json')
    # The saved import camera is resized to the generated 1254px image.
    # All condition masks and original render use the original 1024px camera.
    camera = condition['camera']
    imported_camera = read(SRC / 'anchor_camera.json')
    assert np.allclose(camera['R'], imported_camera['R']) and np.allclose(camera['t'], imported_camera['t'])
    for key, dimension in [('fx', 'width'), ('cx', 'width'), ('fy', 'height'), ('cy', 'height')]:
        assert np.isclose(imported_camera[key] / imported_camera[dimension], camera[key] / camera[dimension])
    rgb = np.asarray(Image.open(SRC / 'anchor.png').convert('RGB'))
    hint = np.asarray(Image.open(SRC / 'position_hint.png').convert('RGB'))
    background_depth = np.asarray(Image.open(SRC / 'depth_background.png').convert('RGB'))
    original_depth = np.asarray(Image.open(SRC / 'depth.png').convert('RGB'))
    generated_native = Image.open(SRC / 'edited_scene.png').convert('RGB')
    height, width = rgb.shape[:2]
    assert (width, height) == (camera['width'], camera['height'])
    generated = np.asarray(generated_native.resize((width, height), Image.Resampling.LANCZOS))
    masks = {n: np.asarray(Image.open(SRC / (n + '.png'))) > 0 for n in [
        'edit_mask', 'old_visible_mask', 'old_clean_mask', 'protected_foreground_mask',
        'extension_proposal', 'generation_bbox_mask']}
    edit = masks['edit_mask']
    guard = masks['protected_foreground_mask'] & masks['extension_proposal']
    assert not (guard & edit).any()
    assert int(guard.sum()) == report['protected_foreground_pixels']
    assert np.array_equal(background_depth[~edit], original_depth[~edit])
    assert not background_depth[edit].any()
    new_world = np.asarray(region['bbox_corners'])
    old_world = np.asarray(m1['bbox_corners'])
    new_xy = project(new_world, camera)
    old_xy = project(old_world, camera)
    projected_bbox = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(projected_bbox, cv2.convexHull(np.round(new_xy).astype(np.int32)), 1)
    assert np.array_equal(projected_bbox.astype(bool), masks['generation_bbox_mask'])
    # Saved corner order is binary x/y/z, with local y up. This is the real
    # lower face of the saved generation region, not a hand-drawn floor patch.
    frame = region['world_transform']
    local = (new_world - frame['center']) @ np.asarray(frame['axes'])
    floor_ids = np.flatnonzero(np.isclose(local[:, 1], min(local[:, 1])))
    assert len(floor_ids) == 4
    floor_xy = new_xy[floor_ids]
    floor_xy = floor_xy[np.argsort(np.arctan2(floor_xy[:, 1] - floor_xy[:, 1].mean(), floor_xy[:, 0] - floor_xy[:, 0].mean()))]
    floor_residual = new_world[floor_ids] @ np.asarray(m1['up_estimation']['normal']) + m1['up_estimation']['offset']
    # Leave exact colors in the saved position hint; add cyan only on the saved
    # protected foreground intersecting the candidate growth region.
    shown_hint = hint.copy()
    shown_hint[guard] = (.40 * hint[guard] + .60 * np.array([0, 222, 231])).astype(np.uint8)
    Image.fromarray(shown_hint).save(OUT / 'analysis/position_hint_with_protection.png')
    Image.fromarray((guard * 255).astype('uint8')).save(OUT / 'analysis/protected_local_mask.png')
    Image.fromarray(generated).save(OUT / 'analysis/generated_on_condition_canvas.png')
    diff = np.abs(generated.astype(np.float32) - rgb.astype(np.float32)).mean(2) / 255.
    np.save(OUT / 'analysis/absolute_rgb_difference.npy', diff)
    metrics = dict(inside_edit_rgb_mae=float(diff[edit].mean()), outside_edit_rgb_mae=float(diff[~edit].mean()),
                   protected_rgb_mae=float(diff[guard].mean()), protected_pixels=int(guard.sum()),
                   edit_pixels=int(edit.sum()), protected_edit_overlap_pixels=int((guard & edit).sum()),
                   difference_definition='mean absolute RGB channel difference in [0,1]; generated RGB resized to 1024px with Lanczos; no registration or color correction',
                   new_support_floor_plane_max_abs_residual=float(np.abs(floor_residual).max()),
                   projection_exactly_matches_saved_generation_bbox_mask=True,
                   depth_outside_edit_byte_identical=True, depth_inside_edit_all_zero=True)
    write(OUT / 'analysis/metrics.json', metrics)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'pdf.fonttype': 42,
                         'axes.linewidth': .5, 'savefig.facecolor': 'white'})
    W, H = 180., 105.
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')

    def axat(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def text(x, y, s, **kw):
        fig.text(x / W, y / H, s, **kw)

    def image(ax, data, crop=None, **kw):
        native_h, native_w = data.shape[:2]
        ax.imshow(data, extent=[-.5, width - .5, height - .5, -.5], interpolation='lanczos', **kw)
        crop = crop or [0, 0, width, height]
        ax.set_xlim(crop[0] - .5, crop[2] - .5)
        ax.set_ylim(crop[3] - .5, crop[1] - .5)
        ax.set_axis_off()

    def wire(ax, points, color, lw=1., style='solid', alpha=1.):
        for i, j in EDGES:
            ax.plot(points[[i, j], 0], points[[i, j], 1], color=color, lw=lw,
                    linestyle=style, alpha=alpha, solid_capstyle='round')

    def legend(handles, x, y, ncol, fontsize=5.8):
        return fig.legend(handles=handles, loc='center', bbox_to_anchor=(x / W, y / H),
                          ncol=ncol, frameon=False, fontsize=fontsize, handlelength=1.3,
                          handletextpad=.4, columnspacing=.8, borderaxespad=0)

    headings = ['(a) 3D Generation\nSpace', '(b) Visibility-Aware\nEdit Region',
                '(c) Background\nDepth', '(d) Generated 2D\nReference']
    xs = [2., 46.5, 91., 135.5]
    for x, title in zip(xs, headings):
        text(x, 101.5, title, va='top', ha='left', fontsize=7.7, fontweight='semibold', linespacing=1.08)
    top = [axat(x, 50., 42.5, 42.5) for x in xs]
    image(top[0], rgb)
    top[0].contour(masks['old_visible_mask'], levels=[.5], colors=[OLD], linewidths=.65)
    wire(top[0], new_xy, MAGENTA, lw=1.05)
    image(top[1], shown_hint)
    top[1].contour(guard, levels=[.5], colors=[CYAN], linewidths=.55)
    gx, gy, ex, ey = GUARD_CROP
    top[1].add_patch(Rectangle((gx, gy), ex-gx, ey-gy, fill=False, edgecolor='white', linewidth=.6, linestyle=(0, (3, 2))))
    image(top[2], background_depth)
    image(top[3], np.asarray(generated_native))
    legend([Line2D([0], [0], color=OLD, lw=1.3, label='M1 target'),
            Line2D([0], [0], color=MAGENTA, lw=1.3, label='3D space')], 23.25, 47.0, 2, 5.6)
    legend([Line2D([0], [0], color=MAGENTA, lw=2.8, label='Edit'),
            Line2D([0], [0], color=CYAN, lw=2.8, label='Protected')], 67.75, 47.0, 2, 5.7)
    text(112.25, 47., 'Black = unknown depth', ha='center', va='center', fontsize=5.9)
    text(156.75, 47., 'Case02 · Roller conveyor', ha='center', va='center', fontsize=5.8)

    text(2, 41., '(e) Workspace Boundary', ha='left', va='center', fontsize=7.5, fontweight='semibold')
    text(50, 41., '(f) Foreground Preservation', ha='left', va='center', fontsize=7.5, fontweight='semibold')
    text(112, 41., '(g) Localized Editing', ha='left', va='center', fontsize=7.5, fontweight='semibold')
    # Geometry-only detail, using the SAME anchor projection as the top row.
    e = axat(2, 7., 45., 29.5)
    e.add_patch(Polygon(floor_xy, closed=True, facecolor=SUPPORT, edgecolor=SUPPORT, linewidth=.9, alpha=.25))
    wire(e, old_xy, OLD, lw=.95, style=(0, (3, 2)))
    wire(e, new_xy, MAGENTA, lw=1.1)
    b = np.concatenate([old_xy, new_xy]); lo = b.min(0); hi = b.max(0)
    margin = .06 * (hi - lo)
    e.set_xlim(lo[0] - margin[0], hi[0] + margin[0]); e.set_ylim(hi[1] + margin[1], lo[1] - margin[1])
    e.set_aspect('equal'); e.set_axis_off()
    legend([Line2D([0], [0], color=OLD, lw=1.1, linestyle='--', label='Old'),
            Line2D([0], [0], color=MAGENTA, lw=1.1, label='New'),
            Line2D([0], [0], color=SUPPORT, lw=2.5, label='Support')], 24.5, 3.9, 3, 5.6)
    for x, label, data in [(50, 'Position hint + guard', shown_hint), (80.5, 'Generated reference', generated)]:
        text(x + 14.25, 36.9, label, ha='center', va='center', fontsize=5.8)
        fa = axat(x, 6.8, 28.5, 28.5)
        image(fa, data, GUARD_CROP)
        # Same mask coordinates on both images, not a new segmentation of output.
        fa.contour(guard, levels=[.5], colors=[CYAN], linewidths=.6)
    text(79.5, 3.9, 'Same foreground rail · same image coordinates', ha='center', va='center', fontsize=5.5)
    gw = 21.2
    diff_ax = None
    for x, label, data in [(112, 'Original', rgb), (134.4, 'Generated', generated), (156.8, '|ΔRGB|', diff)]:
        text(x+gw/2, 36.9, label, ha='center', va='center', fontsize=5.8)
        gh = gw * (CHANGE_CROP[3]-CHANGE_CROP[1]) / (CHANGE_CROP[2]-CHANGE_CROP[0])
        ga = axat(x, 35.3-gh, gw, gh)
        if data.ndim == 2:
            image(ga, data, CHANGE_CROP, cmap='inferno', vmin=0, vmax=.5)
            diff_ax = ga.images[0]
        else:
            image(ga, data, CHANGE_CROP)
    cax = axat(158.0, 10.8, 18.8, 1.4)
    cb = fig.colorbar(diff_ax, cax=cax, orientation='horizontal', ticks=[0, .5])
    cb.ax.set_xticklabels(['0', '0.5+']); cb.ax.tick_params(labelsize=5.3, length=1, pad=1.0); cb.outline.set_visible(False)
    text(112, 11.5, 'Mean |ΔRGB| (full image)', fontsize=5.5, va='center')
    text(112, 6.7, f"Inside: {metrics['inside_edit_rgb_mae']:.3f}   Outside: {metrics['outside_edit_rgb_mae']:.3f}", fontsize=5.5, va='center')
    text(112, 3.1, 'No geometric or color registration', fontsize=5.2, va='center', color='#555555')

    fig.savefig(OUT / 'figure_F11.png', dpi=300)
    fig.savefig(OUT / 'figure_F11.pdf')
    panel_bounds = {name: [x-.3, 44.7, x+42.8, 103.0] for name, x in zip(
        ['a_3d_generation_space', 'b_visibility_aware_edit_region', 'c_background_depth', 'd_generated_2d_reference'], xs)}
    panel_bounds.update(e_workspace_boundary=[1.2, 1.0, 47.8, 43.2],
                        f_foreground_preservation=[49.2, 1.0, 109.8, 43.2],
                        g_localized_editing=[111.2, 1.0, 179., 43.2])
    for name, bounds in panel_bounds.items():
        fig.savefig(OUT/'panels'/f'{name}.png', dpi=600,
                    bbox_inches=Bbox.from_extents(*[v/25.4 for v in bounds]))
    plt.close(fig)
    # Raw local comparisons remain available for inspection at original resolution.
    for key, array, crop in [('f_hint', shown_hint, GUARD_CROP), ('f_generated', generated, GUARD_CROP),
                             ('g_original', rgb, CHANGE_CROP), ('g_generated', generated, CHANGE_CROP)]:
        Image.fromarray(array).crop(crop).save(OUT/'analysis'/f'{key}.png')
    caption = (
        'Fig. 11. Spatial and visibility-aware conditioning for industrial equipment generation (Case02, roller conveyor). '
        '(a) The saved M1 target-Gaussian visible mask (orange contour) and the actual M2 generation volume (magenta wireframe) '
        'are projected into the original 3DGS Anchor rendering. (b) The saved position hint is shown with the local protected '
        'foreground overlaid in cyan for explanation; this cyan annotation and the dashed crop box were not supplied to Qwen. '
        '(c) Original background-depth reference: black pixels inside the edit mask denote unknown depth, and all pixels outside '
        'the mask are retained unchanged from the saved depth image. (d) Previously confirmed Qwen-generated conveyor reference '
        'used in F06. (e) Calibrated projections of the saved M1 target bounding box (orange dashed), M2 generation box (magenta), '
        'and its lower support face (green) show the placement constraints in the same Anchor view; the support is a geometric '
        'annotation, not added floor geometry. (f) The neighboring foreground rail is shown in the input and generated reference, '
        'with the identical saved protection contour superimposed on both. (g) Corresponding workstation crops and absolute RGB '
        'difference; numerical values are mean per-channel absolute differences in [0,1] over the full edit region and its complement. '
        'The heatmap saturates at 0.5. The generated 1254-pixel square is resized to the 1024-pixel conditioning canvas for mask '
        'overlays and differences, without geometric registration or color correction; the main generated-image panel retains '
        'the native source image. The saved Anchor is the existing calibrated observed-pose expanded-canvas view. The actual '
        'output retains scene context but is not pixel-identical outside the edit region: protected input pixels are constraints, '
        'not a guarantee of unchanged output. These are illustrative saved conditions and one actual generated result, not a new '
        'ablation or quantitative superiority experiment. No localization, generation, Gaussian optimization or new 3D rendering was run.')
    (OUT/'caption.txt').write_text(caption+'\n', encoding='utf-8')
    selection = dict(selected_case=CASE, reasons=[
        'Clear original workstation and complete generated roller conveyor',
        'Largest locally protected component is a recognizable adjacent foreground rail',
        '15,436 protected candidate-growth pixels versus 3,017 for Case07',
        'Uses the same confirmed generated source as F06; no substitute or new generation'],
        candidates={k: {name: v[name] for name in ['protected_in_proposal_pixels', 'edit_pixels', 'guard_bounds_xyxy', 'generated_size']} for k,v in manifest.items()})
    write(OUT/'selection_report.json', selection)
    sources = dict(case=CASE, selection=selection, source_files=provenance['source_files'],
        m1_directory=provenance['m1'], m2_directory=provenance['m2'],
        scene_source=m1['source_ply'], source_scene_sha256=condition['source_scene_sha256'],
        source_m1_gaussian_indices_sha256=condition['input_sha256']['target_indices.npy'],
        source_m1_target_gaussians_sha256=condition['input_sha256']['target_gaussians.ply'],
        reference_figure='F06_2d_to_3d/sources.json', camera=camera, imported_output_camera=imported_camera,
        original_m1_reference_camera=m1['reference_camera'], saved_avp_selection=condition.get('avp_selection'),
        full_scene_render='Unchanged existing anchor.png from original 3DGS',
        projected_geometry=dict(generation_corners_world=new_world.tolist(), m1_bbox_corners_world=old_world.tolist(),
            generation_corners_pixels=new_xy.tolist(), m1_bbox_corners_pixels=old_xy.tolist(),
            support_corner_ids=floor_ids.tolist(), support_face_pixels=floor_xy.tolist(),
            floor_plane=m1['up_estimation'], frame=frame),
        display_masks=dict(old='old_visible_mask.png', edit='edit_mask.png',
            cyan='protected_foreground_mask.png AND extension_proposal.png',
            cyan_is_figure_annotation_only=True, masks_modified=False),
        crop_coordinates_1024=dict(foreground=GUARD_CROP, localized_change=CHANGE_CROP),
        display_resize=dict(generated_source=list(generated_native.size), condition_canvas=[width,height],
            interpolation='Lanczos', geometric_registration=False, color_correction=False),
        observed_differences=metrics, figure_size_mm=[W,H], dpi=300, panel_dpi=600,
        qwen_calls=0, trellis_calls=0, training_steps=0, new_gaussian_renders=0,
        limitations=['Saved AVP record uses an expanded observed-pose fallback and records manual review status',
                     'Generation was externally provided; no new API or independent generation claim',
                     'Foreground protection is a supplied spatial condition, not an output pixel-identity guarantee'])
    write(OUT/'sources.json', sources)
    for name, record in provenance['source_files'].items():
        assert sha(SRC/name) == record['sha256'] and sha(OUT/'inputs'/name) == record['sha256']
    write(OUT/'validation.json', dict(source_copies_hash_verified=True, original_files_unchanged=True,
        camera_resize_consistent=True, projection_matches_saved_bbox_mask=True,
        no_edit_protection_overlap=True, depth_unknown_only_inside_edit=True,
        generated_matches_confirmed_f06_source=True, no_experiment_rerun=True))
    shutil.copy2(__file__, OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2000', '-png', str(OUT/'figure_F11.pdf'), str(OUT/'pdf_proof')], check=True)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>F11 Spatial Conditioning</title>'
        '<style>body{max-width:1500px;margin:24px auto;font:16px/1.5 Arial}img{width:100%}</style>'
        '<h1>F11 · Spatial Conditioning · Case02</h1><p>Saved M1/M2 conditions and the confirmed Qwen result. No models rerun.</p>'
        '<p><a href="figure_F11.png">PNG</a> · <a href="figure_F11.pdf">PDF</a> · <a href="panels/">Panels</a> · '
        '<a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a></p><img src="figure_F11.png">', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), selected_case=CASE, metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
