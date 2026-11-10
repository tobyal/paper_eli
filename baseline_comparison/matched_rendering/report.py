"""Assemble completed read-only matched renders into publication/review artifacts."""
import argparse,csv,html,io,json,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison/matched_views')
p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['case00','case02','case04','case07','case10']);a=p.parse_args()
rows=[];sections=[];overview=Image.new('RGB',(2500,90+540*len(a.cases)),'white')
d=ImageDraw.Draw(overview);font=ImageFont.truetype('DejaVuSans.ttf',27)
d.text((16,12),'Viewpoint-Matched Rendering | Independent editing locations',font=font,fill='black')
d.text((16,50),'Original | GaussianEditor | DGE | GSLE | Ours   -   Camera positions may differ across methods.',font=ImageFont.truetype('DejaVuSans.ttf',22),fill='#444444')
for i,case in enumerate(a.cases):
    folder=ROOT/case;r=json.loads((folder/'matching_report.json').read_text());cams=json.loads((folder/'camera_params.json').read_text())
    assert r['input_files_unchanged'] and r['training_steps']==0
    im=Image.open(folder/'comparison_main.png').convert('RGB')
    overview.paste(im.crop((0,80,2500,620)),(0,90+i*540))
    d.text((12,90+i*540+38),case,font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
    for method,data in cams['methods'].items():
        for v in data['views']:
            sel=v['selection'].get('selected',{})
            rows.append(dict(case=case,method=method,view=v['view'],yaw_difference=v['yaw_difference'],pitch_difference=v['pitch_difference'],
                distance=v['distance'],capture=sel.get('capture'),visibility=sel.get('visibility'),
                object_projected_size=v['final_object_projection']['size'],
                target_center=json.dumps(v['target_center']),camera_position=json.dumps(v['camera_position']),
                scene=r['sources'][method]['scene']))
    sections.append(f'<section><h2><a href="{case}/index.html">{case}</a></h2><a href="{case}/comparison_main.png"><img src="{case}/comparison_main.png" loading="lazy"></a><p><a href="{case}/comparison_crops.png">Local crops</a> | <a href="{case}/comparison_multiview.png">Auxiliary views</a> | <a href="{case}/baseline_before_after.png">Baseline before/after</a> | <a href="{case}/camera_params.json">Camera audit</a></p></section>')
name='overview_all_cases.png' if len(a.cases)==5 else 'overview_preview.png'
overview.save(ROOT/name,dpi=(300,300))
ImageOps.contain(overview,(2000,2400)).save(ROOT/'overview_web.jpg',quality=94)
buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows);(ROOT/'camera_summary.csv').write_text(buf.getvalue())
(ROOT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Viewpoint-Matched Rendering</title><style>body{font:17px/1.6 system-ui;max-width:1700px;margin:auto;padding:24px;background:#f4f6f8}section{background:white;padding:18px;margin:20px 0}img{max-width:100%}a{color:#075ead}</style><h1>Viewpoint-Matched Rendering</h1><p>每个方法围绕自己的实际定位/生成区域匹配观察方向、视角和展示尺度。不同方法的相机位置可能不同，这不是同相机像素级比较。Original 与 Ours 使用 Ours 已保存的 Anchor 相机；辅助视角以该相机绕目标左右各转 20 度构建。所有图片来自未改动的最终 PLY，统一 1024×1024 渲染。</p><p>GSLE 使用论文流程复现的原始 DI-SDS 最终结果，不替换为 Coarse 或弱 DI-SDS。Ours 严格使用 prompts.json 中的结果路径；已有 guard 标记保留在记录中。失败、定位泄漏及生成扩散均如实展示。DGE 多区域定位采用其自身原生主视角贡献最大的连通区域作为取景中心，完整定位统计保留在报告中。</p><p><a href="camera_summary.csv">Camera summary CSV</a> | <a href="'''+name+'">Full-resolution overview</a></p>'+''.join(sections)+'</html>',encoding='utf-8')
caption='Viewpoint-Matched Rendering. Each method is rendered from its unchanged final Gaussian scene around its independently localized or generated target. Target instances may differ because localization is independent. Viewing directions and intrinsics follow the Ours reference, with bounded camera-angle and distance adjustments for visibility and framing. Camera positions may differ between methods; these are not pixel-aligned comparisons. Original and Ours share the exact reference camera. The GSLE column uses the paper-derived reproduction with original DI-SDS refinement, including observed failures. Independent render resolution: 1024 x 1024. Figure layout: 300 dpi. Local crops use the same 1.22x padding policy around each method\'s projected target. No Gaussian geometry, opacity, appearance, or scene placement was changed.'
(ROOT/'figure_caption.txt').write_text(caption)
archive=ROOT/'matched_views_review.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['index.html','overview_all_cases.png','overview_preview.png','overview_web.jpg','camera_summary.csv','figure_caption.txt','validation.json','visual_review.json']:
        if (ROOT/name).exists():z.write(ROOT/name,name)
    for case in a.cases:
        for path in (ROOT/case).rglob('*'):
            if path.is_file() and path.suffix in ('.png','.json','.html','.txt'):z.write(path,path.relative_to(ROOT))
print('Packaged',a.cases,archive.stat().st_size,flush=True)
