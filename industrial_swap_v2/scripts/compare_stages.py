"""Arrange saved stage outputs without generation, retouching, or GPU work."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'outputs' / 'comparison'
OUT.mkdir(parents=True, exist_ok=True)
FONT = next((p for p in [Path('C:/Windows/Fonts/msyh.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    if p.is_file()), None)
if FONT is None:
    raise RuntimeError('Install Microsoft YaHei or Noto Sans CJK to draw Chinese labels.')


def font(size):
    return ImageFont.truetype(str(FONT), size)


sources = [
    ('M1  定位阶段', '原场景，尚未替换设备', 'm2/anchor.png'),
    ('M2  二维生成', 'Qwen 生成的参考图', 'm2/edited_scene.png'),
    ('M3  初步三维替换', 'TRELLIS.2 → 3DGS 合成', 'm3/renders/anchor_after.png'),
    ('M4  精细化后', 'DI-SDS 2,000 步', 'm4/renders/anchor_after.png'),
]
images = [Image.open(BASE/'outputs'/row[2]).convert('RGB') for row in sources]
assert all(im.size == (1024, 1024) for im in images)
for stage in ('m3', 'm4'):
    before = np.asarray(Image.open(BASE/'outputs'/stage/'renders/anchor_before.png').convert('RGB'))
    assert np.array_equal(before, np.asarray(images[0])), f'{stage} reference differs'

camera = json.loads((BASE/'outputs/m2/anchor_camera.json').read_text(encoding='utf-8'))
reference = json.loads((BASE/'outputs/m1/manifest.json').read_text(encoding='utf-8'))['reference_camera']
assert np.array_equal(camera['R'], reference['R']) and np.array_equal(camera['t'], reference['t'])
for k in ('fx', 'fy', 'cx', 'cy'):
    assert abs(camera[k]/camera['width']-reference[k]/reference['width']) < 1e-10

# Identical source-pixel crop in every panel: no per-stage alignment or warping.
crop = (290, 270, 720, 800)


def sheet(name, crop_box=None):
    panel_width = 1024 if crop_box is None else 645
    panel_height = 1024 if crop_box is None else 795
    gap, margin, top, header, footer = 18, 24, 90, 116, 70
    width = 2*margin + 4*panel_width + 3*gap
    height = top + header + panel_height + footer
    canvas = Image.new('RGB', (width, height), '#f3f5f7')
    draw = ImageDraw.Draw(canvas)
    title = 'M1 → M4：同一参考视角' if crop_box is None else '设备局部：四张图使用完全相同的裁剪范围'
    draw.text((margin, 18), title, font=font(34), fill='#142332')
    for i, ((label, subtitle, _), im) in enumerate(zip(sources, images)):
        x = margin+i*(panel_width+gap)
        draw.rectangle((x, top, x+panel_width, top+header+panel_height), fill='white')
        draw.text((x+20, top+12), label, font=font(38), fill='#142332')
        draw.text((x+20, top+67), subtitle, font=font(26), fill='#4c6172')
        if crop_box is not None:
            im = im.crop(crop_box).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(im, (x, top+header))
    note = 'Anchor: avp_099 | M2 为二维生成图；M3/M4 为同相机三维渲染。仅排版与等比例缩放，未调色或修补。'
    draw.text((margin, height-footer+19), note, font=font(25 if crop_box is None else 22), fill='#4c6172')
    canvas.save(OUT/name)


sheet('m1_m4_same_view.png')
sheet('m1_m4_same_view_detail.png', crop)
manifest = {
    'camera': camera, 'common_crop_xyxy': crop,
    'original_reference_pixels_identical_in_m3_m4': True,
    'm1_display': 'Original scene at the M1 reference camera; M1 localizes rather than generates RGB.',
    'm2_display': 'Qwen 2D output conditioned on the same original image; not a calibrated 3D render.',
    'operations': ['label', 'side-by-side layout', 'same source-coordinate crop', 'uniform resize for detail panels'],
    'sources': [{'stage': row[0], 'path': row[2], 'sha256': hashlib.sha256((BASE/'outputs'/row[2]).read_bytes()).hexdigest()} for row in sources],
}
(OUT/'comparison_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'outputs': [str(OUT/'m1_m4_same_view.png'), str(OUT/'m1_m4_same_view_detail.png')],
    'same_reference_verified': True}, ensure_ascii=False))
