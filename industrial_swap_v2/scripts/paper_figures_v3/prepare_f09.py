"""Prepare three fixed-condition Qwen requests; no API/model execution."""
import hashlib
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
sys.path.insert(0, str(ROOT))
from industrial_swap_v2.core.m2_prompt import build_prompt

OUT = ROOT/'industrial_swap_v2/outputs/paper_figures_v3/F09_text_variants'
VARIANTS = {
    'B_scara': ('SCARA Industrial Robot', 'A compact blue-and-white four-axis industrial SCARA robot mounted on a low square steel pedestal. It has two horizontal articulated arm links, a vertical spindle, a compact two-finger gripper, and a solid industrial base. The complete robot should fit within the designated replacement space, with clear mechanical connections and no additional equipment.'),
    'C_cnc': ('CNC Machining Center', 'A compact industrial CNC vertical machining center with a rigid light-gray metal enclosure, a large dark-tinted front safety window, a sliding protective door, a side-mounted control panel with a digital display, and a solid rectangular steel base. The machine has realistic industrial proportions and a complete self-contained structure, with no external equipment.'),
    'D_conveyor': ('Industrial Roller Conveyor', 'A compact standalone industrial powered roller conveyor with multiple parallel stainless-steel rollers, two dark-blue metal side frames, a small enclosed drive motor, and four sturdy square steel support legs. The conveyor forms one complete horizontal material transfer unit with realistic mechanical proportions, designed to fit within the designated workstation footprint without extending into surrounding equipment.')}
FILES = ['anchor.png', 'position_hint.png', 'depth_background.png', 'depth.png',
         'edit_mask.png', 'anchor_camera.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    assert not OUT.exists(), 'Preserve existing figure inputs'
    OUT.mkdir(parents=True)
    row = json.loads((ROOT/'baseline_comparison/prompts.json').read_text())['cases'][0]
    assert row['case'] == 'case00'
    source = Path(row['ours_anchor']).parent
    hashes = {f: sha(source/f) for f in FILES}
    m1 = Path(row['target_record']).parent
    target = json.loads((m1/'target.json').read_text())
    edit_region = json.loads((m1/'edit_region.json').read_text())
    summaries = []
    metadata = dict(case='case00', stage='Qwen inputs only; no generated variants B/C/D',
        qwen_calls=0, trellis_calls=0, m3_m4_runs=0,
        fixed_condition_source=str(source), fixed_image_camera_sha256=hashes,
        image_order=FILES[:3], shared_mask_and_camera=True,
        variant_A=dict(status='existing result reused', replacement_text=row['B'],
                       m2_edited_scene=str(source/'edited_scene.png'), final_scene=row['ours_scene']), variants={})
    for key, (label, text) in VARIANTS.items():
        dest = OUT/('variant_'+key);dest.mkdir()
        for name in FILES:
            shutil.copy2(source/name, dest/name)
            assert sha(dest/name) == hashes[name]
        ctx = SimpleNamespace(cfg={'m1': {'target_text': row['A']},
            'm2': {'prompt_mode': 'generic_device', 'replacement_text': text}})
        prompt = build_prompt(ctx)
        (dest/'generation_prompt.txt').write_text(prompt+'\n', encoding='utf-8')
        record = dict(variant=key, title=label, target_text=row['A'], replacement_text=text,
            model='qwen-image-2.1-pro', status='prepared_only', api_calls=0,
            image_order=FILES[:3], inspection_only=FILES[3:],
            source_m1_directory=str(m1), m1_target=target, fixed_edit_region=edit_region,
            source_m2_directory=str(source), source_conditions=json.loads((source/'conditions.json').read_text()),
            fixed_conditions_sha256=hashes, prompt_builder='industrial_swap_v2/core/m2_prompt.py:build_prompt(generic_device)',
            future_generated_image='edited_scene.png', future_import_kind='edited_scene',
            camera_policy='Preserve the entire square anchor canvas. Do not crop or change view.')
        save(dest/'manifest.json', record)
        short=key.split('_',1)[1]
        with zipfile.ZipFile(OUT/f'case00_{short}.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(dest.iterdir()):archive.write(p,p.name)
        metadata['variants'][key] = dict(directory=dest.name, zip=f'case00_{short}.zip',
                                        replacement_text=text, prompt_sha256=sha(dest/'generation_prompt.txt'))
        summaries.append(f'VARIANT {key}: {label}\nTARGET DESCRIPTION\n{text}\n\nFULL QWEN PROMPT\n{prompt}\n')
    (OUT/'prompts_summary.txt').write_text('\n\n'.join(summaries), encoding='utf-8')
    instructions = '''F09: Text-Controllable Industrial Equipment Replacement

STATUS: INPUTS ONLY. No Qwen API, TRELLIS.2, M3, or M4 was run.
Variant A is the existing case00 six-axis robot. B/C/D do not yet have results.

For each variant, upload these THREE images in this order:
1. anchor.png - original RGB
2. position_hint.png - allowed replacement area
3. depth_background.png - surrounding depth; black edit region is UNKNOWN
Paste the complete generation_prompt.txt into Qwen-Image-2.1-Pro.
depth.png, edit_mask.png and anchor_camera.json are inspection/handoff metadata.
All six shared condition/camera files are byte-identical across B/C/D and to
the existing case00 M2. Only target descriptions and the prompt text change.
The old case00 projected-volume mask is intentionally reused rather than
recomputing variant-dependent spaces. Stay within that original footprint.

Save each returned full-scene RGB as edited_scene.png in its variant directory
(or supply all three files with their variant names). Do not crop or change the
camera. The aspect ratio must remain 1:1. Preserve the original result files.

NEXT STAGE (not executed): create a fresh experiment for each variant, reusing
case00 M1 and these exact M2 conditions; import the full-scene image through the
existing Web image-upload/asset-library path, or through:
  python -m industrial_swap_v2.scripts.import_case_image --config NEW_CONFIG.yaml
    --image /path/to/edited_scene.png --kind edited_scene
NEW_CONFIG.yaml must point to the fresh experiment with prepared M2 conditions
and the corresponding replacement text. This existing importer disables Qwen,
runs foreground extraction, and stops before M3/M4. Do not point it at case00's
completed M2 directory. After review, M3/M4 can be launched separately.
'''
    (OUT/'README.txt').write_text(instructions,encoding='utf-8')
    save(OUT/'input_manifest.json',metadata)
    canvas=Image.new('RGB',(1800,1280),'white');d=ImageDraw.Draw(canvas)
    bold=ImageFont.truetype('DejaVuSans-Bold.ttf',32);font=ImageFont.truetype('DejaVuSans.ttf',23)
    d.text((30,22),'F09 | Fixed scene conditions, three target descriptions',font=bold,fill='black')
    for n,(f,label) in enumerate(zip(FILES[:3],['1. Anchor RGB','2. Position Hint','3. Background Depth'])):
        d.text((30+n*590,90),label,font=bold,fill='black')
        im=Image.open(source/f).convert('RGB');im.thumbnail((550,550),Image.Resampling.LANCZOS)
        canvas.paste(im,(30+n*590,140))
    y=725
    for key,(label,text) in VARIANTS.items():
        d.text((30,y),f'Variant {key[0]} - {label}',font=bold,fill='black');y+=46
        lines=textwrap.wrap(text,width=128)
        for line in lines:d.text((30,y),line,font=font,fill='#303030');y+=32
        y+=20
    canvas=canvas.crop((0,0,1800,y+25));canvas.save(OUT/'input_overview.png',dpi=(300,300))
    with zipfile.ZipFile(OUT/'case00_all_variants.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':archive.write(p,p.relative_to(OUT))
    for name,h in hashes.items():assert sha(source/name)==h
    print('F09 COMPLETE:',OUT,flush=True)


if __name__=='__main__':main()
