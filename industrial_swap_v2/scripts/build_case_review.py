"""Build honest, labeled contact sheets from actual pre-Qwen case artifacts.

Runs on CPU locally or remotely. No image generation, retouching or API calls.
Original photographs and Gaussian-rendered Anchor views are labeled separately.
"""
import argparse
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def font_path():
    for path in ('C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():return path
    raise FileNotFoundError('Set --font to a font supporting Chinese and English')


def wrap(text,font,width):
    lines=[]
    for paragraph in text.split('\n'):
        current=''
        for character in paragraph:
            if font.getlength(current+character)>width and current:
                # Prefer word boundaries in English; Chinese remains character-wrapped.
                split=current.rfind(' ')
                if split>len(current)*.65:
                    lines.append(current[:split]);current=current[split+1:]+character
                else:lines.append(current);current=character
            else:current+=character
        lines.append(current)
    return lines


def paste_contained(canvas,path,box):
    x,y,w,h=box
    image=Image.open(path).convert('RGB');image.thumbnail((w,h),Image.Resampling.LANCZOS)
    canvas.paste(image,(x+(w-image.width)//2,y+(h-image.height)//2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--font');args=parser.parse_args();batch=args.batch.resolve()
    font=args.font or font_path();large=ImageFont.truetype(font,32);medium=ImageFont.truetype(font,23)
    small=ImageFont.truetype(font,20);tiny=ImageFont.truetype(font,17)
    rows=json.loads((batch/'cases_summary_rows.json').read_text(encoding='utf-8'))
    assert len(rows)==10
    if (batch/'visual_review_notes.json').exists():
        notes=json.loads((batch/'visual_review_notes.json').read_text(encoding='utf-8'))['cases']
        for row in rows:row['quality_notes']=notes.get(row['case_id'],row['quality_notes'])
        (batch/'cases_summary_rows.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    panels=[('真实照片 / 定位参考相机','localization/source_photo.jpg'),
        ('真实照片 + 自动 SAM 参考分割','localization/source_photo_overlay.jpg'),
        ('Anchor RGB / 原始高斯场景渲染','m2/anchor.png'),
        ('Anchor 旧设备可见 Mask','m1/old_device_mask.png'),
        ('Anchor 三维定位高斯贡献叠加','m1/localization_overlay.png'),
        ('删除目标高斯后的背景 / 未补全','m1/background_render.png'),
        ('Qwen 输入 2 / 三维位置提示','m2/position_hint.png'),
        ('Qwen 输入 3 / 背景深度（黑色未知）','m2/depth_background.png'),
        ('原场景深度 / 仅检查，不提交','m2/depth.png')]
    summary_panels=[('Anchor RGB','m2/anchor.png'),('定位叠加','m1/localization_overlay.png'),
        ('删除后背景','m1/background_render.png'),('位置提示','m2/position_hint.png'),('背景深度','m2/depth_background.png')]
    overview=Image.new('RGB',(2040,2060),'#edf1f5');od=ImageDraw.Draw(overview)
    od.text((24,14),'10 个工业设备替换案例 · M1–M2 生成前检查',font=large,fill='#162a43')
    od.text((24,60),'实际语义查询与三维分割；未调用 Qwen / TRELLIS，未执行 M3 / M4。全部待人工确认。',font=medium,fill='#334155')
    htmlcards=[]
    for index,row in enumerate(rows):
        folder=batch/row['case_id'];case=json.loads((folder/'case.json').read_text(encoding='utf-8'))
        prompt=(folder/'m2/generation_prompt.txt').read_text(encoding='utf-8')
        textblocks=[('M1 semantic: '+row['m1_english_semantic'],medium),
            ('Replacement: '+row['replacement_prompt'],medium),
            ('检查备注：'+row['quality_notes'],small),
            ('提交顺序：m2/anchor.png → m2/position_hint.png → m2/depth_background.png。深度黑区是未知，不是补好的地面。',small),
            ('完整 Qwen Prompt（尚未提交）：\n'+prompt,small)]
        wrapped=[(wrap(text,f,1744),f) for text,f in textblocks]
        textheight=sum(len(lines)*(f.size+8)+12 for lines,f in wrapped)
        canvas=Image.new('RGB',(1800,124+3*465+textheight+25),'white');draw=ImageDraw.Draw(canvas)
        draw.text((28,16),row['case_id']+'  '+row['original_device'],font=large,fill='#162a43')
        stats=f"Target: {row['target_gaussian_count']:,} Gaussians | Anchor: {row['anchor_view']} | visibility {row['anchor_visibility']:.1%} | SAM reference IoU {row['reference_sam_iou']:.3f}"
        draw.text((28,64),stats,font=small,fill='#334155')
        draw.text((28,92),'IoU 为自动 SAM 一致性指标，不是人工真值精度。红色为目标高斯的可见贡献。',font=small,fill='#7b4b00')
        for pi,(label,relative) in enumerate(panels):
            x=pi%3*600;y=124+pi//3*465
            draw.text((x+18,y+9),label,font=small,fill='#1e293b')
            paste_contained(canvas,folder/relative,(x+12,y+42,576,414))
        y=124+3*465+10
        for lines,f in wrapped:
            for line in lines:draw.text((28,y),line,font=f,fill='#263449');y+=f.size+8
            y+=12
        canvas.save(folder/'review_sheet.jpg',quality=95)
        # Full camera comparisons retain the same render camera in each row.
        views=case['review_views'];multi=Image.new('RGB',(1600,60+len(views)*380),'white');md=ImageDraw.Draw(multi)
        md.text((16,12),row['case_id']+'  多视角定位与删除检查',font=large,fill='#162a43')
        for vi,view in enumerate(views):
            viewdir=folder/'m1/review_views'/Path(view['folder']).name
            for ci,name in enumerate(('rgb','old_mask','overlay','background')):
                x=ci*400;y=60+vi*380;md.text((x+8,y+4),view['camera']['name']+' / '+name,font=tiny,fill='#334155')
                paste_contained(multi,viewdir/(name+'.png'),(x+5,y+32,390,340))
        multi.save(folder/'multiview_review.jpg',quality=94)
        x=20+(index%2)*1020;y=110+(index//2)*390
        od.rounded_rectangle((x,y,x+1000,y+374),radius=8,fill='white')
        od.text((x+12,y+8),row['case_id']+'  '+row['original_device']+f"  |  {row['target_gaussian_count']:,} Gaussians",font=medium,fill='#162a43')
        for li,line in enumerate(wrap(row['m1_english_semantic'],small,976)):
            od.text((x+12,y+44+li*25),line,font=small,fill='#334155')
        for pi,(label,relative) in enumerate(summary_panels):
            sx=x+8+pi*198;od.text((sx+5,y+98),label,font=tiny,fill='#334155')
            paste_contained(overview,folder/relative,(sx,y+125,190,190))
        od.text((x+12,y+324),f"Anchor {row['anchor_view']}  |  visible {row['anchor_visibility']:.1%}  |  SAM IoU {row['reference_sam_iou']:.2f}  |  待人工确认",font=tiny,fill='#7b4b00')
        rel=row['case_id'];imgs=''.join(f'<a href="{rel}/{p}"><img src="{rel}/{p}" alt="{html.escape(label)}"><span>{html.escape(label)}</span></a>' for label,p in summary_panels)
        htmlcards.append(f'''<article><h2>{html.escape(rel+' '+row['original_device'])}</h2>
<p><b>M1:</b> {html.escape(row['m1_english_semantic'])}</p><div class="strip">{imgs}</div>
<p><b>新设备：</b>{html.escape(row['replacement_prompt'])}</p><p>{html.escape(row['quality_notes'])}</p>
<p><a href="{rel}/review_sheet.jpg">完整九图检查表</a> · <a href="{rel}/multiview_review.jpg">同相机多视角检查</a> · <a href="{rel}/m2/generation_prompt.txt">完整 Qwen Prompt</a></p>
<details><summary>Qwen 输入与 Prompt</summary><p>按顺序上传 Anchor RGB、位置提示图、背景深度图。不要上传九图检查表。</p><pre>{html.escape(prompt)}</pre></details></article>''')
    overview.save(batch/'cases_overview.jpg',quality=96)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>10 个案例：生成前检查</title>
<style>body{font:16px system-ui,sans-serif;background:#edf1f5;color:#263449;max-width:1500px;margin:30px auto;padding:0 20px}h1,h2{color:#162a43}article{background:white;padding:20px;margin:24px 0;border-radius:10px}.strip{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.strip img{width:100%}.strip span{display:block}a{color:#1858a1}pre{white-space:pre-wrap;line-height:1.6}p{line-height:1.6}</style>
<h1>10 个工业设备替换案例 · 生成前检查</h1><p>以下均为实际 M1 定位与 M2 条件。尚未调用 Qwen / TRELLIS，也未执行 M3 / M4。IoU 是对自动 SAM 参考的比较，不是人工真值。黑色深度区表示未知；删除后背景没有补全。</p>
<p><a href="cases_overview.jpg">十例总览</a> · <a href="cases_summary.csv">CSV 清单</a> · <a href="README.txt">文件说明与后续导入方法</a></p>'''+''.join(htmlcards)+'</html>'
    (batch/'index.html').write_text(page,encoding='utf-8')
    print('Built ten review sheets, ten multiview sheets, one overview, and a static review index.')


if __name__=='__main__':main()
