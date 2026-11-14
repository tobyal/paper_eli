"""Replace only F11 panel (a), preserving all other figure content.

Render panel on the server with Matplotlib; assemble PNG/PDF using the local
bundled pypdf runtime. No Gaussian, model, localization or training reruns.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
PREVIOUS = BASE/'F11_spatial_conditioning_final_v2'
OUT = BASE/'F11_spatial_conditioning_final_v3'
W, H, DPI = 180., 146., 300
# Original panel allocation remains fixed; only its content changes.
PANEL_MM = [1., 52.5, 72.5, 145.]
CROP = [192, 270, 832, 910]
ORANGE, MAGENTA, GREEN = '#EFAD36', '#D42BCE', '#70B494'
GAIN = .48
EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(PREVIOUS)): sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}


def render_panel():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Patch
    from matplotlib.transforms import Bbox

    before = snapshot()
    for sub in ['inputs', 'analysis', 'panels', 'code']:
        shutil.copytree(PREVIOUS/sub, OUT/sub, dirs_exist_ok=True)
    inputs = OUT/'inputs'
    camera = read(inputs/'anchor_camera.json')
    previous_sources = read(PREVIOUS/'sources.json')
    assert camera == previous_sources['camera']
    assert CROP[1::2] == previous_sources['crops']['condition'][1::2]
    rgb = np.asarray(Image.open(inputs/'anchor.png').convert('RGB'))
    visible = np.load(inputs/'target_visible_alpha.npy')
    region = read(inputs/'generation_region.json')
    assert np.isfinite(visible).all()
    assert np.array_equal(visible > region['settings']['old_alpha_threshold'],
                          np.asarray(Image.open(inputs/'old_visible_mask.png')) > 0)
    color = np.asarray(matplotlib.colors.to_rgb(ORANGE))*255
    alpha = GAIN*np.clip(visible, 0, 1)[..., None]
    colored = np.rint(rgb*(1-alpha)+color*alpha).astype('uint8')
    Image.fromarray(colored).save(OUT/'analysis/a_localized_gaussians_full.png')

    corners = np.asarray(region['bbox_corners'])
    camera_points = corners @ np.asarray(camera['R']).T + camera['t']
    assert np.all(camera_points[:, 2] > 0)
    projected = camera_points[:, :2]/camera_points[:, 2:] * [camera['fx'], camera['fy']] + [camera['cx'], camera['cy']]
    assert np.allclose(projected, previous_sources['changes']['m2_projected_corners'], atol=1e-7)
    support_ids = [0, 1, 5, 4]
    for dimension in range(2):
        assert projected[:, dimension].min() > CROP[dimension]
        assert projected[:, dimension].max() < CROP[dimension+2]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
    fig.text(2/W, 142.3/H, '(a) 3D Workspace Construction', fontsize=7.5, fontweight='semibold', va='center')
    a = fig.add_axes([2/W, 68.5/H, 69.5/W, 69.5/H])
    a.imshow(colored, extent=[-.5, 1023.5, 1023.5, -.5], interpolation='lanczos')
    a.set_xlim(CROP[0]-.5, CROP[2]-.5)
    a.set_ylim(CROP[3]-.5, CROP[1]-.5)
    a.set_xticks([]); a.set_yticks([])
    for spine in a.spines.values():
        spine.set_visible(False)
    a.add_patch(Polygon(projected[support_ids], closed=True, facecolor=GREEN,
                        edgecolor=GREEN, alpha=.38, linewidth=1.1))
    for i, j in EDGES:
        a.plot(projected[[i, j], 0], projected[[i, j], 1], color=MAGENTA,
               lw=1.4, solid_capstyle='round')
    # No M1 contour, M1 box, dimensions, or spatial inset is drawn.
    for y, handle, label in [
        (65.4, Patch(facecolor=ORANGE, alpha=.6), 'Localized Equipment'),
        (61.9, Line2D([0], [0], color=MAGENTA, lw=1.4), 'Generation Space'),
        (58.4, Patch(facecolor=GREEN, alpha=.6), 'Support Plane')]:
        fig.legend(handles=[handle], labels=[label], bbox_to_anchor=(2/W, y/H),
                   loc='center left', frameon=False, fontsize=6.2, borderaxespad=0,
                   handlelength=1.6, handletextpad=.6)
    fig.savefig(OUT/'analysis/a_layer_full_canvas.png', dpi=DPI)
    bounds = Bbox.from_extents(*[v/25.4 for v in PANEL_MM])
    fig.savefig(OUT/'analysis/a_replacement_vector.pdf', bbox_inches=bounds)
    fig.savefig(OUT/'panels/a_3d_workspace_construction.png', dpi=600, bbox_inches=bounds)
    plt.close(fig)
    write(OUT/'analysis/a_revision.json', dict(
        previous=str(PREVIOUS), panel_bounds_mm=PANEL_MM, crop=CROP,
        original_a_crop=previous_sources['crops']['workspace'],
        b_crop_unchanged=previous_sources['crops']['condition'],
        same_anchor_camera_as_b=True, same_vertical_crop_as_b=True,
        horizontal_crop_note='Centered subset of B to preserve the existing square A layout without stretching',
        selected_gaussian_contribution=dict(file='inputs/target_visible_alpha.npy', sha256=sha(inputs/'target_visible_alpha.npy'),
            definition='Cached full-scene transmittance-weighted M1 selected-Gaussian contribution',
            display_gain=GAIN, color=ORANGE, binary_fill=False, stored_parameters_changed=False),
        geometry=dict(source='inputs/generation_region.json', sha256=sha(inputs/'generation_region.json'),
            world_corners=corners.tolist(), projected_corners=projected.tolist(),
            support_ids=support_ids, magenta_line_width_pt=1.4,
            support_fill_alpha=.38, support_line_width_pt=1.1,
            old_box_drawn=False, target_contour_drawn=False, inset_drawn=False),
        camera=camera, camera_sha256=sha(inputs/'anchor_camera.json'),
        old_source_files_unchanged=before == snapshot(), gaussian_renders=0,
        model_calls=0, train_steps=0))
    assert before == snapshot()
    print('Rendered only replacement panel A.', flush=True)


def assemble():
    from pypdf import PdfReader, PdfWriter, Transformation
    import pypdfium2 as pdfium

    before = snapshot()
    old_png = PREVIOUS/'figure_F11_final.png'
    old_pdf = PREVIOUS/'figure_F11_final.pdf'
    base_image = Image.open(old_png).convert('RGBA')
    layer = Image.open(OUT/'analysis/a_layer_full_canvas.png').convert('RGBA')
    assert layer.size == base_image.size
    x0, y0, x1, y1 = PANEL_MM
    pixel_box = (int(np.floor(x0*DPI/25.4)), int(np.floor((H-y1)*DPI/25.4)),
                 int(np.ceil(x1*DPI/25.4)), int(np.ceil((H-y0)*DPI/25.4)))
    result = base_image.copy()
    result.paste(layer.crop(pixel_box), pixel_box[:2])
    result.save(OUT/'figure_F11_final.png', dpi=(DPI, DPI))
    changed = np.any(np.asarray(base_image) != np.asarray(result), axis=2)
    allowed = np.zeros(changed.shape, bool)
    allowed[pixel_box[1]:pixel_box[3], pixel_box[0]:pixel_box[2]] = True
    assert not (changed & ~allowed).any()

    # Overlay only the small A rectangle. Other PDF content stays vector-native
    # and is reused directly from the preceding PDF, not re-plotted or rasterized.
    original_reader = PdfReader(old_pdf)
    panel_reader = PdfReader(OUT/'analysis/a_replacement_vector.pdf')
    page, panel = original_reader.pages[0], panel_reader.pages[0]
    assert abs(float(panel.mediabox.width)-(x1-x0)*72/25.4) < .01
    assert abs(float(panel.mediabox.height)-(y1-y0)*72/25.4) < .01
    page.merge_transformed_page(panel, Transformation().translate(x0*72/25.4, y0*72/25.4), over=True)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'F11 spatial conditioning - revised panel A',
                         '/Subject': 'Only panel A modified; B-D retained unchanged'})
    with (OUT/'figure_F11_final.pdf').open('wb') as f:
        writer.write(f)

    def pdf_pixels(path, scale=2):
        doc = pdfium.PdfDocument(str(path))
        page = doc[0]
        bitmap = page.render(scale=scale)
        pixels = np.asarray(bitmap.to_pil().convert('RGB')).copy()
        bitmap.close(); page.close(); doc.close()
        return pixels

    proof_old = pdf_pixels(old_pdf)
    proof_new = pdf_pixels(OUT/'figure_F11_final.pdf')
    assert proof_old.shape == proof_new.shape
    # Allow two boundary pixels for PDF antialiasing; full B-D remain outside.
    factor = 2*72/25.4
    rect = [max(0, int(np.floor(x0*factor))-2), max(0, int(np.floor((H-y1)*factor))-2),
            int(np.ceil(x1*factor))+2, int(np.ceil((H-y0)*factor))+2]
    pdf_changed = np.any(proof_old != proof_new, axis=2)
    pdf_allowed = np.zeros(pdf_changed.shape, bool)
    pdf_allowed[rect[1]:rect[3], rect[0]:rect[2]] = True
    assert not (pdf_changed & ~pdf_allowed).any(), 'PDF content changed outside panel A'
    Image.fromarray(pdf_pixels(OUT/'figure_F11_final.pdf', scale=3)).save(OUT/'pdf_proof.png')

    original_caption = (PREVIOUS/'caption.txt').read_text(encoding='utf-8')
    a_caption = (
        '(a) Orange semitransparent highlighting uses the cached continuous visible contribution of '
        'the actual M1-selected Gaussians, preserving their occlusion and opacity distribution. '
        'The magenta generation volume and green support face use the unchanged saved M2 geometry. '
        'The old localization box, target contour and inset are omitted. '
        'The Anchor camera and vertical crop match (b); a narrower horizontal crop preserves the square panel. ')
    caption = re.sub(r'\(a\).*?(?=\(b\))', a_caption, original_caption, count=1)
    (OUT/'caption.txt').write_text(caption, encoding='utf-8')
    revision = read(OUT/'analysis/a_revision.json')
    write(OUT/'sources.json', dict(
        case='case00', revision_scope='Panel A only', previous_sources=read(PREVIOUS/'sources.json'),
        previous_png_sha256=sha(old_png), previous_pdf_sha256=sha(old_pdf),
        panel_a=revision, camera=revision['camera'],
        other_panels_reused_byte_identically=True,
        figure_png_outside_a_pixel_identical=True, figure_pdf_outside_a_pixel_identical=True,
        identical_b_c_d_layout=True, canvas_mm=[W,H], png_dpi=DPI,
        generated_from_current_adaptive_conditions=False,
        qwen_calls=0, trellis_calls=0, train_steps=0, gaussian_renders=0))
    untouched_panels = {}
    for p in (PREVIOUS/'panels').iterdir():
        if p.name != 'a_3d_workspace_construction.png' and p.is_file():
            untouched_panels[p.name] = sha(p) == sha(OUT/'panels'/p.name)
    assert all(untouched_panels.values())
    assert before == snapshot()
    write(OUT/'validation.json', dict(
        old_version_preserved=True, only_a_pixels_changed=True,
        png_changed_pixels_outside_a=int((changed & ~allowed).sum()),
        pdf_changed_pixels_outside_a=int((pdf_changed & ~pdf_allowed).sum()),
        untouched_panel_files=untouched_panels, original_gaussians_and_geometry_unchanged=True,
        no_old_box_no_contour_no_inset=True, genuine_continuous_gaussian_contribution=True,
        same_camera_as_b=True, same_vertical_crop_as_b=True))
    shutil.copy2(__file__, OUT/'code'/Path(__file__).name)
    (OUT/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>F11 - final A highlight</title>'
        '<style>body{font:16px/1.5 Arial;max-width:1500px;margin:24px auto}img{max-width:100%}</style>'
        '<p>Only panel A updated; B-D preserved pixel-identically.</p>'
        '<p><a href="figure_F11_final.png">PNG</a> | <a href="figure_F11_final.pdf">PDF</a> | '
        '<a href="panels/">Panels</a> | <a href="sources.json">Sources</a></p>'
        '<img src="figure_F11_final.png">', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), outside_a_changed_pixels=0,
                         other_panels_byte_identical=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['render', 'assemble'], required=True)
    args = parser.parse_args()
    render_panel() if args.stage == 'render' else assemble()
