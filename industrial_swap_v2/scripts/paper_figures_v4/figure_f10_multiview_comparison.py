"""Revise confirmed F10 with paired 3DGS renders; no scene/model updates.

CUDA_VISIBLE_DEVICES=1 envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f10_multiview_comparison.py --render
python3 industrial_swap_v2/scripts/paper_figures_v4/figure_f10_multiview_comparison.py --assemble
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
CONFIRMED = ROOT / 'experiment figure/F10'
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F10_multiview_comparison'
PANEL_SIZE = 1024
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source(path):
    path = Path(path)
    return {'path': str(path), 'resolved': str(path.resolve()),
            'sha256': sha(path), 'bytes': path.stat().st_size}


def render():
    import torch
    sys.path.insert(0, str(ROOT / 'gsle'))
    import common as c
    assert not (OUT / 'render_sources.json').exists(), 'Preserve completed renders; use --assemble for layout only.'
    for name in ('inputs', 'panels', 'renders', 'code'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    # The confirmed collection, not an older experimental figure, selects all cameras/crops.
    prior = read(CONFIRMED / 'sources.json')
    assert len(prior['selection']) == 5
    pinned = {}
    for name in ('sources.json', 'caption.txt', 'delivery_manifest.json'):
        dst = OUT / 'inputs' / ('previous_F10_' + name)
        shutil.copy2(CONFIRMED / name, dst)
        pinned[name] = source(CONFIRMED / name)
        assert sha(dst) == pinned[name]['sha256']
    old_scene = prior['original_sources']['scene']
    m4 = Path(old_scene['path']).parent
    original_path = ROOT / 'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    guarded_path = m4 / 'scene_candidate_step2000.ply'
    composition = read(m4 / 'candidate_composition.json')
    summary = read(m4 / 'background_optimization/summary.json')
    sources = {'original': source(original_path), 'reconfigured': source(guarded_path),
               'object': source(m4 / 'object_world.ply'),
               'composition': source(m4 / 'candidate_composition.json'),
               'guard_summary': source(m4 / 'background_optimization/summary.json')}
    m1 = read(m4.parent / 'm1/manifest.json')
    assert sources['original']['sha256'] == m1['source_sha256']
    assert sources['reconfigured']['sha256'] == composition['scene_sha256']
    assert sources['object']['sha256'] == composition['object_sha256']
    assert sources['object']['sha256'] == prior['original_sources']['object']['sha256']
    assert summary['object_all_parameters_bitwise_preserved']
    views = []
    for selection in prior['selection']:
        record = selection['original_camera_record']
        camera = record['camera']
        crop = selection['existing_display_crop']
        x0, y0, x1, y1 = crop
        assert x1 - x0 == y1 - y0
        assert 0 <= x0 < x1 <= camera['width'] and 0 <= y0 < y1 <= camera['height']
        ratio = PANEL_SIZE / (x1 - x0)
        effective = dict(camera)
        effective.update(width=PANEL_SIZE, height=PANEL_SIZE,
                         fx=camera['fx'] * ratio, fy=camera['fy'] * ratio,
                         cx=(camera['cx'] - x0) * ratio, cy=(camera['cy'] - y0) * ratio)
        views.append({'view': selection['view'], 'label': selection['label'],
                      'camera': camera, 'crop_xyxy': crop, 'panel_camera': effective,
                      'old_selected_row': selection['selected_original_row'],
                      'yaw_offset_degrees': record['yaw_offset_degrees'],
                      'pitch_offset_degrees': record['pitch_offset_degrees'],
                      'camera_and_crop_unchanged_from_confirmed': True,
                      'same_camera_both_rows': True, 'panel_size': [PANEL_SIZE, PANEL_SIZE],
                      'panels': {k: f'panels/{selection["view"]}_{k}.png' for k in ('original', 'reconfigured')}})
    torch.cuda.set_per_process_memory_fraction(.35)
    panel_sources = {}
    with torch.no_grad():
        for state, path in [('original', original_path), ('reconfigured', guarded_path)]:
            model = c.GaussianData.load(path)
            panel_sources[state] = {}
            for view in views:
                camera = c.Camera.from_record(view['camera'])
                print('Render', state, view['view'], flush=True)
                rgb = c.render(camera, model, aux=False)['rgb']
                full_path = OUT / 'renders' / f'{view["view"]}_{state}_full.png'
                c.save_image(full_path, rgb)
                panel = OUT / view['panels'][state]
                Image.open(full_path).crop(view['crop_xyxy']).resize(
                    (PANEL_SIZE, PANEL_SIZE), LANCZOS).save(panel, dpi=(300, 300))
                panel_sources[state][view['view']] = {'full_render': source(full_path), 'panel': source(panel)}
                del rgb
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert all(sha(Path(r['path'])) == r['sha256'] for r in sources.values())
    assert all(sha(CONFIRMED / name) == r['sha256'] for name, r in pinned.items())
    write(OUT / 'camera_manifest.json', {
        'case': 'case00', 'views': views,
        'coordinates': 'COLMAP world-to-camera, +x right, +y down, +z forward',
        'renderer': 'gsle/common.py Gaussian rasterizer, stored SH appearance, scale_modifier=1, black background',
        'same_render_settings_for_both_rows': True,
        'panel_resampling': 'Saved crop, then identical LANCZOS resize to 1024x1024; no stretch.',
        'source': pinned['sources.json'], 'source_snapshot': 'inputs/previous_F10_sources.json',
        'camera_changes': 0, 'crop_changes': 0,
    })
    write(OUT / 'render_sources.json', {
        'basis': 'experiment figure/F10: user-confirmed five-view selection',
        'prior_confirmed_inputs': pinned, 'previous_scene': old_scene,
        'sources': sources, 'panels': panel_sources,
        'scene_selection': {
            'policy': 'User-selected dual-mask optimized Step-2000 scene; do not substitute the guarded Step-0 fallback.',
            'guarded_selected_step': summary['selected_step'],
            'iterations_in_existing_run': summary['iterations'],
            'displayed_candidate': True,
            'displayed_step': 2000,
            'note': 'The displayed dual-mask result is the existing Step-2000 candidate. Its original guard rejection remains recorded.',
        },
        'no_real_photographs': True, 'no_gaussian_modifications': True,
        'generation_calls': 0, 'training_steps': 0, 'input_sha256_unchanged': True,
    })


def assemble():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('PaperSans', str(fonts / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('PaperBold', str(fonts / 'DejaVuSans-Bold.ttf')))
    cameras = read(OUT / 'camera_manifest.json')
    provenance = read(OUT / 'render_sources.json')
    width = 180 / 25.4 * 72
    margin, row_label_width, gap, row_gap, heading = 3.5, 58., 3., 5., 22.
    cell = (width - 2 * margin - row_label_width - gap - 4 * gap) / 5
    height = 2 * margin + heading + 2 * cell + row_gap
    pdf = OUT / 'figure_F10_revised.pdf'
    cv = canvas.Canvas(str(pdf), pagesize=(width, height), pageCompression=1)
    cv.setTitle('Original and reconfigured industrial scenes from matched cameras')
    cv.setAuthor('paper_eli')
    cv.setFillColorRGB(1, 1, 1)
    cv.rect(0, 0, width, height, stroke=0, fill=1)
    xstart = margin + row_label_width + gap
    top_bottom = margin + cell + row_gap
    labels = [('(a) Front View',), ('(b) Left Oblique', 'View'),
              ('(c) Right-side', 'View'), ('(d) Rear View',), ('(e) Elevated View',)]
    cv.setFillColorRGB(.08, .08, .08)
    cv.setFont('PaperBold', 7.25)
    for i, lines in enumerate(labels):
        x = xstart + i * (cell + gap)
        for j, text in enumerate(lines):
            assert pdfmetrics.stringWidth(text, 'PaperBold', 7.25) < cell
            cv.drawCentredString(x + cell / 2, height - margin - 8.5 - j * 8.4, text)
    for state, y, words in [('original', top_bottom, ['Original', '3DGS']),
                            ('reconfigured', margin, ['Reconfigured', '3DGS'])]:
        cv.setFont('PaperBold', 7.0)
        for j, text in enumerate(words):
            assert pdfmetrics.stringWidth(text, 'PaperBold', 7.0) < row_label_width
            cv.drawCentredString(margin + row_label_width / 2, y + cell / 2 + 2.3 - j * 9, text)
        for i, view in enumerate(cameras['views']):
            cv.drawImage(str(OUT / view['panels'][state]), xstart + i * (cell + gap), y,
                         width=cell, height=cell, preserveAspectRatio=True)
    cv.save()
    subprocess.run(['pdftoppm', '-r', '300', '-singlefile', '-png', str(pdf), str(OUT / 'figure_F10_revised')], check=True)
    png = OUT / 'figure_F10_revised.png'
    image = Image.open(png).convert('RGB')
    image.save(png, dpi=(300, 300))
    image.thumbnail((1900, 1900), LANCZOS)
    image.save(OUT / 'preview.jpg', quality=95)
    scene_note = (
        'The reconfigured row uses the saved dual-mask optimized 2,000-step scene '
        '(scene_candidate_step2000.ply), matching the previously confirmed F10 scene. '
        'This is the existing optimization candidate requested for display; the original '
        'environment guard rejected it and the separate guarded output retained Step 0. '
        'The candidate is not relabeled as guard-accepted. '
        if provenance['scene_selection']['displayed_candidate'] else
        'The reconfigured scene is the saved guarded scene_final.ply, with its background retained at Step 0. '
    )
    caption = (
        'Fig. 10. Matched-camera multi-view comparison before and after industrial equipment '
        'replacement in Case00. The top row renders the original industrial 3DGS; the bottom '
        'row renders the same factory after replacing the localized machine with the stored '
        'yellow six-axis robot Gaussian. Columns retain the five cameras and crops from the '
        'confirmed F10: (a) front, (b) left oblique, (c) right side, (d) rear, and (e) elevated. '
        'These are relative viewing-direction labels, not absolute factory coordinates. '
        'Camera extrinsics, intrinsics, rasterizer settings, crop bounds, resolution, and '
        'display resampling are identical within every before/after pair. The front, rear '
        'and elevated views preserve their confirmed closer crops; the oblique and right-side '
        'views retain the wider factory context. All images are real Gaussian renders, including '
        'existing reconstruction artifacts and foreground occlusions; no photographs, masks '
        'or image retouching are substituted. ' + scene_note +
        'The equipment geometry, placement, colors and all '
        'scene Gaussian parameters are unchanged. No model generation, registration or '
        'optimization was rerun.\n'
    )
    (OUT / 'caption.txt').write_text(caption, encoding='utf-8')
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    layout = {'rows': 2, 'columns': 5, 'width_mm': 180, 'height_mm': height / 72 * 25.4,
              'panel_mm': cell / 72 * 25.4, 'png_dpi': 300, 'png_pixels': list(Image.open(png).size),
              'header_font_pt': 7.25, 'row_label_font_pt': 7.0, 'pdf_text_vector': True,
              'no_insets_masks_or_charts': True}
    write(OUT / 'sources.json', dict(**provenance, cameras=source(OUT / 'camera_manifest.json'),
                                   layout=layout, script=source(Path(__file__))))
    write(OUT / 'validation.json', {
        'camera_pairs_identical': all(v['same_camera_both_rows'] for v in cameras['views']),
        'old_cameras_and_crops_unchanged': True,
        'panel_count': len(list((OUT / 'panels').glob('*.png'))),
        'all_panels_same_size': all(Image.open(p).size == (PANEL_SIZE, PANEL_SIZE) for p in (OUT / 'panels').glob('*.png')),
        'original_experiment_files_unchanged': provenance['input_sha256_unchanged'],
        'png': source(png), 'pdf': source(pdf),
    })
    (OUT / 'README.txt').write_text(
        'F10: matched-camera original/reconfigured 3DGS comparison, based on experiment figure/F10.\n'
        'Five existing cameras and crops; ten 1024x1024 panels; full renders retained.\n'
        + scene_note + '\nNo experiments rerun.\n' +
        'camera_manifest.json: exact render and cropped-display cameras for both rows.\n'
        'inputs/: previous confirmed F10 metadata; code/: reproducible rendering/layout script.\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'layout': layout}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--render', action='store_true')
    p.add_argument('--assemble', action='store_true')
    args = p.parse_args()
    assert args.render or args.assemble
    if args.render:
        render()
    if args.assemble:
        assemble()
