"""Prepare isolated same-camera M2 comparisons; generation calls are forbidden."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import yaml
from ..core.context import Context,read_json,write_json,sha256,gsle
from ..core.m2_space import VERSION,DEFAULTS,generation_frame
from ..modules.m2_image_generation import run


def panel(images,titles,path):
    size=768; header=86
    canvas=Image.new('RGB',(size*2,(size+header)*2),'white'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
    for i,(im,title) in enumerate(zip(images,titles)):
        x=(i%2)*size; y=(i//2)*(size+header)
        draw.multiline_text((x+15,y+10),title,font=font,fill='#19212b',spacing=5)
        canvas.paste(im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS),(x,y+header))
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keys',nargs='+',default=['case_01_A','case_01_B','case_01_C','case_02_B','case_03_A'])
    a=p.parse_args();root=Path.cwd();base=root/'industrial_swap_v2';dest=a.output.resolve();dest.relative_to(base)
    dest.mkdir(parents=True,exist_ok=False);batch=read_json(a.batch);summaries=[]
    for key in a.keys:
        item=next(x for x in batch['items'] if x['key']==key)
        old=Path(item['output_directory']);case=root/'industrial_swap_v2/webapp/data/m1_assist/cases'/item['case_id']
        watched=[case/'integrity.json',*[old/n for n in item['image_sha256']]]
        before={str(path):sha256(path) for path in watched}
        for name,digest in read_json(case/'integrity.json').items():assert sha256(case/name)==digest
        folder=dest/key;folder.mkdir();outputs=folder/'outputs';outputs.mkdir()
        (outputs/'m1').symlink_to(case/'outputs/m1',target_is_directory=True)
        cfg=yaml.safe_load((old.parents[1]/'config.yaml').read_text())
        cfg['project'].update(gpu=3,output_dir=str((outputs).relative_to(base)),
            records_dir=str((folder/'records').relative_to(base)),logs_dir=str((folder/'logs').relative_to(base)))
        cfg['m2'].update(reuse_anchor=True,existing_anchor=str(old/'anchor_camera.json'),
            spatial_conditions=dict(DEFAULTS),preparation_only=True)
        cfg['m2'].pop('prompt_override',None)
        cfg['m3']['input_image']=str((outputs/'m2/object_rgba.png').relative_to(base))
        cp=folder/'config.yaml';cp.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        ctx=Context(cp)
        with patch('industrial_swap_v2.modules.m2_image_generation.generate',side_effect=AssertionError('Qwen forbidden in validation')):
            run(ctx,prepare_only=True)
        out=outputs/'m2';conditions=read_json(out/'conditions.json');report=read_json(out/'mask_report.json')
        region=read_json(out/'generation_region.json');camera=read_json(out/'anchor_camera.json')
        assert camera==read_json(old/'anchor_camera.json')
        assert generation_frame(ctx)==region['world_transform']
        images={n:np.asarray(Image.open(out/n).convert('RGB')) for n in ('anchor.png','position_hint.png','depth.png','depth_background.png')}
        mask=np.asarray(Image.open(out/'edit_mask.png'))>127
        assert not (images['depth_background.png'][mask]).any()
        np.testing.assert_array_equal(images['depth_background.png'][~mask],images['depth.png'][~mask])
        np.testing.assert_array_equal(images['position_hint.png'][~mask],images['anchor.png'][~mask])
        np.testing.assert_array_equal(images['anchor.png'],np.asarray(Image.open(old/'anchor.png').convert('RGB')))
        assert not (out/'edited_scene.png').exists() and not (outputs/'m3').exists() and not (outputs/'m4').exists()
        assert not list((folder/'records').glob('qwen*lock'))
        assert before=={str(path):sha256(path) for path in watched}
        for name,digest in read_json(case/'integrity.json').items():assert sha256(case/name)==digest
        oldmask=np.asarray(Image.open(old/'edit_mask.png'))>127
        box=np.asarray(Image.open(out/'generation_bbox_mask.png'))>127
        original=images['anchor.png'];boxim=original.copy();boxim[box]=(.8*boxim[box]+.2*np.array([255,0,255])).astype(np.uint8)
        from ..scripts.project_m1_m2_regions import wireframe,boundary
        c=gsle(ctx);cam=c.Camera.from_record(camera);xy,z=cam.project(np.asarray(region['bbox_corners']))
        wireframe(boxim,xy,(248,66,220));boundary(boxim,box,(248,66,220),3)
        panel([Image.open(old/'position_hint.png'),Image.fromarray(boxim),Image.open(out/'position_hint.png'),Image.open(out/'depth_background.png')],
            [f'{key} | 原整框编辑区域\n{oldmask.sum():,} px',
             f'新三维生成框投影（不直接作为编辑 Mask）\n{box.sum():,} px | {region["semantic_profile"]["name"]}',
             f'新可见性约束编辑 Mask\n{mask.sum():,} px | 保留可见贡献 {100*report["visible_alpha_mass_retained"]:.1f}%',
             '新背景深度：仅编辑 Mask 内未知\n相机、RGB、Mask 外的深度均保持一致'],folder/'comparison.png')
        row={'key':key,'camera':camera['name'],'semantic_profile':region['semantic_profile'],
            'old_mask_pixels':int(oldmask.sum()),'new_mask_pixels':int(mask.sum()),'new_bbox_pixels':int(box.sum()),
            'edit_area_reduction':float(1-mask.sum()/oldmask.sum()),'mask_report':report,
            'old_sides':region['geometry']['old_side_lengths'],'new_sides':region['geometry']['new_side_lengths'],
            'same_camera_and_anchor_pixels':True,'m1_and_previous_outputs_unchanged':True,
            'depth_and_hint_unchanged_outside_mask':True,'m3_reads_exact_m2_frame':True,
            'output':str(out),'comparison':str(folder/'comparison.png'),'qwen_called':False}
        summaries.append(row);write_json(dest/'validation.json',summaries)
        print(key,{'old':int(oldmask.sum()),'new':int(mask.sum()),'retained_mass':report['visible_alpha_mass_retained'],
                       'protected_overlap':report['protected_mask_overlap_pixels']},flush=True)
    print('M2 same-camera validation complete:',dest,flush=True)


if __name__=='__main__':main()
