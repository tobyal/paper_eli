"""Correct F09's stale box mask using the current M2 geometry-only preparation.

All three text variants share case00's camera and the newly prepared replacement
space, so this controlled input experiment varies only the prompt. No API or
model-generation entry point is imported. Archives contain 3 PNGs + 1 prompt.
"""
import copy
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256
from industrial_swap_v2.core.depth import prepare_conditions
from industrial_swap_v2.core.m2_prompt import build_prompt
from prepare_f09 import VARIANTS

BASE=ROOT/'industrial_swap_v2'
OUT=BASE/'outputs/paper_figures_v3/F09_text_variants'
DEST=OUT/'qwen_direct_adaptive_v2'
NAMES={'anchor.png':'1_anchor.png','position_hint.png':'2_position_hint.png',
       'depth_background.png':'3_background_depth.png'}


def main():
    assert not DEST.exists(),'Preserve previously prepared corrections'
    DEST.mkdir(parents=True)
    work=DEST/'preparation';work.mkdir()
    row=read_json(ROOT/'baseline_comparison/prompts.json')['cases'][0]
    assert row['case']=='case00'
    m1=Path(row['target_record']).parent
    old_m2=Path(row['ours_anchor']).parent
    cfg=yaml.safe_load((BASE/'configs/default.yaml').read_text())
    cfg['project'].update(output_dir=str(work.relative_to(BASE)),
        records_dir=str((work/'records').relative_to(BASE)),logs_dir=str((work/'logs').relative_to(BASE)),gpu=3)
    cfg['m1'].update(target_text=row['A'],localization_dir=str(m1))
    cfg['m2'].update(reuse_anchor=True,existing_anchor=row['ours_anchor'],anchor_resolution=1024,
        replacement_text=row['B'],prompt_mode='generic_device',preparation_only=True,
        max_api_requests=0)
    cfg['m2'].pop('api_key_file',None)
    config_path=work/'config.yaml';config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Read-only source symlink; no M1 execution.
    (work/'m1').symlink_to(m1,target_is_directory=True)
    ctx=Context(config_path);torch.cuda.set_per_process_memory_fraction(.4)
    shared=prepare_conditions(ctx)
    condition=read_json(shared/'conditions.json')
    assert condition['spatial_method']=='adaptive_visible_v1'
    assert condition['camera']==read_json(old_m2/'anchor_camera.json')
    assert sha256(shared/'edit_mask.png')!=sha256(old_m2/'edit_mask.png')
    mask=np.asarray(Image.open(shared/'edit_mask.png'))>0
    depth=np.asarray(Image.open(shared/'depth.png'))
    background=np.asarray(Image.open(shared/'depth_background.png'))
    assert np.array_equal(depth[~mask],background[~mask]) and not background[mask].any()
    oldmask=np.asarray(Image.open(old_m2/'edit_mask.png'))>0
    records={}
    for key,(title,text) in VARIANTS.items():
        dest=DEST/key;dest.mkdir()
        for old,new in NAMES.items():shutil.copy2(shared/old,dest/new)
        prompt_cfg=copy.deepcopy(cfg);prompt_cfg['m2']['replacement_text']=text
        prompt=build_prompt(SimpleNamespace(cfg=prompt_cfg))
        (dest/'prompt.txt').write_text(prompt+'\n',encoding='utf-8')
        archive=DEST/f'case00_{key}_qwen_v2.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(dest.iterdir()):z.write(p,p.name)
        records[key]=dict(title=title,archive=archive.name,prompt=text,
            hashes={p.name:sha256(p) for p in dest.iterdir()})
    combined=DEST/'case00_all_qwen_adaptive_v2.zip'
    with zipfile.ZipFile(combined,'w',zipfile.ZIP_DEFLATED) as z:
        for key in VARIANTS:
            for p in sorted((DEST/key).iterdir()):z.write(p,f'{key}/{p.name}')
    with zipfile.ZipFile(combined) as z:
        assert len(z.namelist())==12 and z.testzip() is None
        assert all(Path(n).name in list(NAMES.values())+['prompt.txt'] for n in z.namelist())
    # Diagnostics remain outside all user input ZIPs.
    canvas=Image.new('RGB',(1800,690),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    panels=[(old_m2/'position_hint.png','Previous: projected-box mask'),
            (shared/'position_hint.png','Updated: visible + depth-constrained'),
            (shared/'depth_background.png','Updated background depth')]
    for i,(p,label) in enumerate(panels):
        im=Image.open(p).convert('RGB');im.thumbnail((580,580))
        canvas.paste(im,(i*600+10,55));draw.text((i*600+10,15),label,font=font,fill='black')
    draw.text((10,650),f'Mask pixels: {oldmask.sum():,} -> {mask.sum():,}. Same M1 and Anchor camera. No Qwen request.',font=font,fill='black')
    canvas.save(DEST/'mask_update_preview.jpg',quality=95)
    record=dict(status='prepared_only',spatial_method=condition['spatial_method'],
        old_m2=str(old_m2),new_m2=str(shared),source_m1=str(m1),
        same_anchor=True,shared_new_conditions_across_variants=True,
        fixed_space_basis=row['B'],fixed_space_policy='Prepare current M2 space once for existing case00, then vary only B/C/D text.',
        old_mask_pixels=int(oldmask.sum()),new_mask_pixels=int(mask.sum()),
        mask_report=condition['mask_report'],background_depth_verified=True,
        qwen_calls=0,trellis_calls=0,training_steps=0,variants=records,
        combined_archive=combined.name,archive_sha256=sha256(combined),
        supersedes=['qwen_direct','case00_scara.zip','case00_cnc.zip','case00_conveyor.zip','case00_all_variants.zip'])
    write_json(DEST/'preparation_record.json',record)
    print(json.dumps({k:record[k] for k in ['new_m2','old_mask_pixels','new_mask_pixels','mask_report','archive_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
