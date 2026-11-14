"""Layout-only, five-panel landscape F05 using approved Case00 images.

Run with remote system python3. No renderer, model, or optimization is loaded.
"""
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

from figure_f05_revised import ROOT, V3, V4, sha, record, dump, wrapped, DISPLAY_A, DISPLAY_B

SOURCE = V4 / 'F05_revised'
OUT = V4 / 'F05_landscape'
PAN = OUT / 'panels'
W, M, GAP, LABEL_H, ROW_GAP = 720., 6., 8., 17., 10.
TOP_IMAGE_W, TOP_H = 164., 123.
A_W = 364.
BOTTOM_W = 350.
BOTTOM_CROP = (0, 360, 1600, 1260)
BOTTOM_H = BOTTOM_W * 9 / 16
H = 2*M + 2*LABEL_H + TOP_H + ROW_GAP + BOTTOM_H
TITLES = ['(a) Reference Image & Text', '(b) Target Localization',
          '(c) Generated 2D Reference', '(d) Original 3DGS', '(e) Reconfigured 3DGS']


def input_panel(c, x, y):
    c.setFillColorRGB(1,1,1)
    c.rect(x, y, A_W, TOP_H, fill=1, stroke=0)
    p = SOURCE / 'panels/a_registered_reference.png'
    iw, ih = Image.open(p).size
    ph = TOP_IMAGE_W * ih / iw
    c.drawImage(str(p), x, y+(TOP_H-ph)/2, TOP_IMAGE_W, ph)
    tx = x + TOP_IMAGE_W + GAP
    tw = A_W - TOP_IMAGE_W - GAP
    top = y + TOP_H
    c.setFillColorRGB(.08,.08,.08)
    c.setFont('PaperBold', 9.8)
    c.drawString(tx, top-11, 'Text A - Original Object')
    wrapped(c, DISPLAY_A, tx, top-27, tw, 9.8, 12.)
    c.setFont('PaperBold', 9.8)
    c.drawString(tx, top-65, 'Text B - Replacement Object')
    last = wrapped(c, DISPLAY_B, tx, top-81, tw, 9.8, 12.)
    assert last >= y-8, 'Input text overflow'


def main():
    PAN.mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE/'sources.json').read_text())
    protected = list(SOURCE.rglob('*'))
    protected += [p for base in [V3, V4] for p in base.glob('*/figure_*') if p.parent != OUT]
    protected = sorted(set(p for p in protected if p.is_file()))
    before = {str(p): sha(p) for p in protected}
    mapping = [
        ('b_target_localization', 'c_target_localization', None),
        ('c_generated_2d_reference', 'd_generated_2d_reference', None),
        ('d_original_3dgs', 'e_original_3dgs_scene', BOTTOM_CROP),
        ('e_reconfigured_3dgs', 'f_reconfigured_3dgs_scene', BOTTOM_CROP),
    ]
    panel_records = {}
    for output_name, input_name, crop in mapping:
        p = SOURCE/'panels'/(input_name+'.png')
        dst = PAN/(output_name+'.png')
        im = Image.open(p).convert('RGB')
        native_size = list(im.size)
        if crop:
            im = im.crop(crop)
            im.save(dst, dpi=(300,300))
        else:
            shutil.copy2(p,dst)
        assert Image.open(dst).convert('RGB').tobytes() == im.tobytes()
        panel_records[output_name] = dict(source=record(p), crop_pixels=crop or [0,0,*native_size],
            size=list(im.size), exact_source_pixels=True, output=record(dst))

    # The combined input panel is separately exported using the same vector layout.
    temporary = OUT/'_input_panel.pdf'
    pc = canvas.Canvas(str(temporary),pagesize=(A_W,TOP_H))
    input_panel(pc,0,0)
    pc.showPage(); pc.save()
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(temporary),
                    str(PAN/'a_reference_image_and_text')],check=True)
    temporary.unlink()

    pdf = OUT/'figure_F05_landscape.pdf'
    c = canvas.Canvas(str(pdf),pagesize=(W,H),pageCompression=1)
    c.setTitle('Reference-image-assisted text-guided industrial equipment replacement')
    c.setAuthor('paper_eli - existing Case00 results, layout only')
    top_y = H-M-LABEL_H-TOP_H
    bottom_y = M
    positions = [
        (M,top_y,A_W,TOP_H),
        (M+A_W+GAP,top_y,TOP_IMAGE_W,TOP_H),
        (M+A_W+GAP*2+TOP_IMAGE_W,top_y,TOP_IMAGE_W,TOP_H),
        (M,bottom_y,BOTTOM_W,BOTTOM_H),
        (M+BOTTOM_W+GAP,bottom_y,BOTTOM_W,BOTTOM_H),
    ]
    names = ['a_reference_image_and_text']+[x[0] for x in mapping]
    for i,(name,(x,y,w,h),label) in enumerate(zip(names,positions,TITLES)):
        c.setFillColorRGB(.08,.08,.08)
        c.setFont('PaperSans',10.2)
        assert pdfmetrics.stringWidth(label,'PaperSans',10.2) <= w
        c.drawString(x,y+h+5,label)
        if i == 0:
            input_panel(c,x,y)
        else:
            c.drawImage(str(PAN/(name+'.png')),x,y,w,h)
    c.showPage(); c.save()
    png = OUT/'figure_F05_landscape.png'
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(pdf),str(png.with_suffix(''))],check=True)
    im = Image.open(png)
    im.save(png,dpi=(300,300))
    ImageOps.contain(im,(1800,1200)).save(OUT/'figure_F05_landscape_preview.jpg',quality=95)

    caption = ('Fig. 5. Reference-image-assisted text-guided industrial equipment replacement. '
        '(a) An unannotated registered factory photograph and text descriptions specifying the original '
        'equipment (A) and its desired replacement (B). (b) Localized target Gaussians highlighted in '
        'translucent red. (c) The saved Qwen-generated 2D replacement reference. '
        '(d) The original industrial 3DGS scene and (e) the reconfigured scene, displayed with '
        'exactly the same camera, field of view, rendering settings, and pixel crop. '
        'The lower row uses a shared landscape display crop of the previously approved renders; '
        'the complete replacement device remains visible. All images are existing Case00 results. '
        'The scene in (e) retains the same saved 2,000-step optimization candidate as the original '
        'F05/F06/F10, rather than the automatic guarded export. No generation, optimization, '
        'retouching, or model modification was performed.\n')
    (OUT/'caption.txt').write_text(caption,encoding='utf-8')
    source['previous_layout_manifest'] = record(SOURCE/'sources.json')
    source['panels'] = panel_records
    source['panels']['a_reference_image_and_text'] = dict(
        photograph=record(SOURCE/'panels/a_registered_reference.png'),
        photograph_crop='none; full unannotated registered image',
        display_text_A=DISPLAY_A,display_text_B=DISPLAY_B,
        output=record(PAN/'a_reference_image_and_text.png'))
    source['render_settings']['e_f_crop'] = list(BOTTOM_CROP)
    source['render_settings']['d_e_shared_crop'] = list(BOTTOM_CROP)
    source['render_settings']['scene_panel_labels'] = ['d','e']
    source['layout'] = dict(rows=2,panels_per_row=[3,2],width_points=W,height_points=H,
        aspect_ratio=W/H,pixels=list(im.size),dpi=300,font='DejaVu Sans embedded vector text',
        no_group_titles=True,no_overall_title=True,top_image_width=TOP_IMAGE_W,
        bottom_image_width=BOTTOM_W,lower_image_area_vs_top=BOTTOM_W*BOTTOM_H/(TOP_IMAGE_W*TOP_H),
        positions=[dict(name=n,box_points=list(p)) for n,p in zip(names,positions)])
    source['script'] = record(Path(__file__))
    dump(OUT/'sources.json',source)
    after = {str(p):sha(p) for p in protected}
    assert before==after,'Previous figure modified'
    dump(OUT/'validation.json',dict(protected_files_unchanged=True,protected_count=len(protected),
        protected_sha256=before,figure_png=record(png),figure_pdf=record(pdf),
        source_pixels_or_declared_crop_exact=True,panel_count=5,
        lower_scene_camera_and_crop_identical=True,models_or_experiments_rerun=False))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>F05 Landscape</title>
<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}</style>
<h1>F05 · 横向两排版</h1><p>上排：照片与 A/B 文本、目标定位、二维参考；下排：严格同相机、同裁剪的三维替换前后对照。</p>
<p><a href="figure_F05_landscape.png">300 DPI PNG</a> · <a href="figure_F05_landscape.pdf">PDF</a> · <a href="../F05_landscape.zip">完整包</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">来源记录</a></p>
<img src="figure_F05_landscape_preview.jpg">''',encoding='utf-8')
    with ZipFile(V4/'F05_landscape.zip','w',ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(output=str(OUT),pixels=list(im.size),aspect_ratio=W/H,
                         lower_scene_crop=list(BOTTOM_CROP),protected_files=len(protected))))


if __name__ == '__main__':
    main()
