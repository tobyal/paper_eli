"""Re-layout the approved Case00 F05 panels; no GPU/model/scene changes.

Run with system python3 from the remote project root. Output is independent
of every original figure. The image pixels come only from saved F05 assets.
"""
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / 'industrial_swap_v2/outputs/paper_figures_v3'
V4 = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4'
OLD = V3 / 'F05_case00'
OUT = V4 / 'F05_revised'
PAN = OUT / 'panels'
WIDTH = 518.4
MARGIN = 8.
GAP = 10.
CELL = (WIDTH - 2 * MARGIN - GAP) / 2
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('PaperSans', str(FONT_DIR / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('PaperBold', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    p = Path(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def wrapped(c, text, x, baseline, width, size=9.4, leading=13.):
    words = text.split()
    lines = []
    line = ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if pdfmetrics.stringWidth(candidate, 'PaperSans', size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    c.setFont('PaperSans', size)
    for line in lines:
        c.drawString(x, baseline, line)
        baseline -= leading
    return baseline


DISPLAY_A = '"Industrial machine with a black assembly on a white base."'
DISPLAY_B = '"Yellow six-axis industrial robotic arm with articulated joints, a steel base, and a two-finger gripper."'


def instructions(c, x, y, w, h):
    c.setFillColorRGB(1, 1, 1)
    c.rect(x, y, w, h, fill=1, stroke=0)
    left = x + 16
    c.setFillColorRGB(.10, .10, .10)
    c.setFont('PaperBold', 9.3)
    c.drawString(left, y + h - 18, 'Original Object (A)')
    wrapped(c, DISPLAY_A, left, y + h - 35, w - 32)
    c.setFont('PaperBold', 9.3)
    c.drawString(left, y + h - 82, 'Replacement Object (B)')
    last = wrapped(c, DISPLAY_B, left, y + h - 99, w - 32, 9.4, 12.5)
    assert last >= y - 2, 'Text instructions overflow'


def main():
    PAN.mkdir(parents=True, exist_ok=True)
    previous = json.loads((OLD / 'sources.json').read_text())
    protected = sorted(p for d in [OLD, V3/'F06_five_cases', V3/'F07_comparison',
                                  V3/'F09_text_variants', V3/'F10_multiview', V4/'F09_text_control']
                       if d.exists() for p in d.glob('figure_*') if p.is_file())
    before = {str(p): sha(p) for p in protected}
    photo = OLD / 'panels/reference_photo_original.jpg'
    qwen = OLD / 'panels/c_qwen_reference.png'
    assert sha(photo) == previous['registered_photo']['sha256']
    assert sha(qwen) == previous['qwen']['sha256']

    specs = [
        ('a_registered_reference', photo, None),
        ('c_target_localization', OLD/'panels/b_target_localization.png', (0, 200, 1600, 1400)),
        ('d_generated_2d_reference', qwen, (0, 128, 1024, 896)),
        ('e_original_3dgs_scene', OLD/'panels/d_original_3dgs.png', None),
        ('f_reconfigured_3dgs_scene', OLD/'panels/e_reconfigured_scene.png', None),
    ]
    panel_sources = {}
    for name, path, crop in specs:
        im = Image.open(path).convert('RGB')
        origsize = list(im.size)
        if crop:
            im = im.crop(crop)
        output = PAN / (name + '.png')
        im.save(output, dpi=(300, 300))
        panel_sources[name] = dict(input=record(path), output=record(output),
                                  input_size=origsize, crop_pixels=crop or [0, 0, *origsize],
                                  output_size=list(im.size), image_content_edited=False)

    photo_size = Image.open(photo).size
    heights = [CELL * photo_size[1] / photo_size[0], CELL * .75, CELL]
    row_header = 32.
    row_gap = 14.
    height = 2*MARGIN + sum(heights) + 3*row_header + 2*row_gap
    labels = [
        ['(a) Registered Reference Image', '(b) Text Instructions'],
        ['(c) Target Localization', '(d) Generated 2D Reference'],
        ['(e) Original 3DGS Scene', '(f) Reconfigured 3DGS Scene'],
    ]
    names = [
        ['a_registered_reference', 'b_text_instructions'],
        ['c_target_localization', 'd_generated_2d_reference'],
        ['e_original_3dgs_scene', 'f_reconfigured_3dgs_scene'],
    ]
    sections = ['Inputs', 'Intermediate Results', '3D Scene Replacement']
    pdf = OUT / 'figure_F05_revised.pdf'
    c = canvas.Canvas(str(pdf), pagesize=(WIDTH, height), pageCompression=1)
    c.setTitle('Reference-image-assisted text-guided industrial equipment replacement')
    c.setAuthor('paper_eli: existing Case00 experimental results')
    cursor = height - MARGIN
    placements = []
    for row in range(3):
        c.setFillColorRGB(.13, .13, .13)
        c.setFont('PaperBold', 9.)
        c.drawString(MARGIN, cursor - 8, sections[row])
        label_y = cursor - 23
        bottom = cursor - row_header - heights[row]
        for col in range(2):
            x = MARGIN + col * (CELL + GAP)
            c.setFont('PaperSans', 8.5)
            title = labels[row][col]
            assert pdfmetrics.stringWidth(title, 'PaperSans', 8.5) < CELL
            c.drawString(x, label_y, title)
            if row == 0 and col == 1:
                instructions(c, x, bottom, CELL, heights[row])
            else:
                p = PAN / (names[row][col] + '.png')
                c.drawImage(str(p), x, bottom, CELL, heights[row])
            placements.append(dict(panel=names[row][col], x=x, y=bottom,
                                   width=CELL, height=heights[row]))
        cursor = bottom - row_gap
    c.showPage()
    c.save()

    # Export the identical vector text panel independently at 300 dpi.
    text_pdf = OUT / '_text_panel_working.pdf'
    tc = canvas.Canvas(str(text_pdf), pagesize=(CELL, heights[0]))
    instructions(tc, 0, 0, CELL, heights[0])
    tc.showPage()
    tc.save()
    subprocess.run(['pdftoppm', '-singlefile', '-r', '300', '-png', str(text_pdf),
                    str(PAN/'b_text_instructions')], check=True)
    text_pdf.unlink()
    panel_sources['b_text_instructions'] = dict(display_text_A=DISPLAY_A,
                                               display_text_B=DISPLAY_B,
                                               output=record(PAN/'b_text_instructions.png'))
    subprocess.run(['pdftoppm', '-singlefile', '-r', '300', '-png', str(pdf),
                    str(OUT/'figure_F05_revised')], check=True)
    png = OUT / 'figure_F05_revised.png'
    im = Image.open(png)
    im.save(png, dpi=(300, 300))
    ImageOps.contain(im, (1600, 2400)).save(OUT/'figure_F05_revised_preview.jpg', quality=95)

    caption = (
        'Fig. 5. Reference-image-assisted text-guided industrial equipment replacement. '
        '(a) A registered reference photograph of the original factory scene, shown without annotations. '
        '(b) Text descriptions specifying the original equipment and the desired replacement. '
        '(c) Localized target Gaussians highlighted in translucent red within the original 3DGS scene. '
        '(d) The saved text-guided 2D replacement reference. '
        '(e) The original 3DGS scene. '
        '(f) The reconfigured 3DGS scene rendered from exactly the same camera, field of view, '
        'full-image bounds, and Gaussian rendering settings as (e). '
        'Panels (c) and (d) use vertical display crops to emphasize the workstation; '
        '(e) and (f) retain the complete, matched original renders. '
        'The scene in (f) is the same saved 2,000-step candidate used by the original F05, F06, and F10; '
        'it is not the automatic guarded export. No generation, optimization, or scene modification '
        'was performed for this revised figure.\n')
    (OUT/'caption.txt').write_text(caption, encoding='utf-8')
    trace = sum(previous['reference_camera']['R'][i][j] * previous['main_camera']['R'][i][j]
                for i in range(3) for j in range(3))
    angle = math.degrees(math.acos(max(-1., min(1., (trace-1)/2))))
    source = dict(case='case00', source_figure=record(OLD/'sources.json'),
                  text_A=previous['old_text'], text_B=previous['new_text'],
                  registered_photo=previous['registered_photo'],
                  localization_indices=previous['localization_indices'],
                  m1_target=previous['m1_target'], qwen=previous['qwen'],
                  original_scene=previous['original_scene'], replacement_scene=previous['final_scene'],
                  replacement_object=previous['final_object'],
                  registered_photo_camera=previous['reference_camera'],
                  localization_render_camera=previous['main_camera'],
                  localization_vs_photo_orientation_difference_degrees=angle,
                  matched_original_replacement_camera=previous['main_camera'],
                  render_settings=dict(reused_native_renders=True, resolution=[1600,1600],
                    e_f_crop=[0,0,1600,1600], e_f_camera_identical=True,
                    e_f_fov_identical=True, e_f_render_parameters_identical=True,
                    original_renderer='baseline_comparison.matched_rendering.render.c.render',
                    background='original renderer default',
                    original_script=record(ROOT/'industrial_swap_v2/scripts/paper_figures_v3/render_ours.py')),
                  localization_annotation='Original target colors blended with 60% red before joint rasterization; original geometry/opacity retained. Same saved F05 rendering.',
                  panels=panel_sources,
                  layout=dict(rows=3,columns=2,width_points=WIDTH,height_points=height,
                    image_row_heights_points=heights,third_row_area_vs_second=heights[2]/heights[1],
                    placements=placements,font='DejaVu Sans embedded vector labels',dpi=300,
                    pixels=list(im.size), no_overall_title=True, no_novel_view=True),
                  previous_automatic_guard_passed=previous['previous_automatic_guard_passed'],
                  qwen_calls=0,trellis_runs=0,optimization_steps=0,scene_ply_modified=False,
                  script=record(Path(__file__)))
    dump(OUT/'sources.json',source)
    after = {str(p): sha(p) for p in protected}
    assert before == after, 'An existing figure changed'
    # Assert all photographic/rendered panels are exact decoded source pixels or declared crops.
    for name, path, crop in specs:
        expected = Image.open(path).convert('RGB')
        if crop:
            expected = expected.crop(crop)
        assert Image.open(PAN/(name+'.png')).tobytes() == expected.tobytes()
    dump(OUT/'validation.json', dict(old_figures_unchanged=before,
         source_pixels_or_declared_crops_exact=True, figure_png=record(png), figure_pdf=record(pdf),
         panel_count=len(list(PAN.glob('*.png'))), original_final_cameras_and_framing_identical=True))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>F05 Revised</title>
<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}p{line-height:1.5}</style>
<h1>F05 · Case00 设备替换流程</h1><p>注册照片与文本 → 定位与二维参考 → 同相机三维替换。复用原版真实结果，未重新生成或优化。</p>
<p><a href="figure_F05_revised.png">300 DPI PNG</a> · <a href="figure_F05_revised.pdf">PDF</a> · <a href="caption.txt">英文图注</a> · <a href="sources.json">来源与相机</a></p>
<img src="figure_F05_revised_preview.jpg"><p>第三行沿用原版的 2,000 步场景候选，与 F06/F10 一致；自动 Guard 状态记录在 sources.json。</p>''',encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), pixels=list(im.size),
                          row_height_ratios=[round(v/CELL,3) for v in heights],
                          protected_figures=len(protected), orientation_difference_degrees=angle)))


if __name__ == '__main__':
    main()
