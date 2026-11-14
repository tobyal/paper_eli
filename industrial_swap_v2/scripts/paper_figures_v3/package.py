"""Validate source immutability, publish a local review index, and package figures."""
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v3'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    folders={'F05':'F05_case00','F06':'F06_five_cases','F07':'F07_comparison','F10':'F10_multiview'}
    figures={};checked={}
    def check(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                p=Path(value['path'])
                if p.is_file() and str(p) not in checked:
                    assert sha(p)==value['sha256'],f'Source changed: {p}'
                    checked[str(p)]=value['sha256']
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    for key,folder in folders.items():
        d=OUT/folder
        for name in ['figure_'+key+'.png','figure_'+key+'.pdf','caption.txt']:
            assert (d/name).is_file(),d/name
        im=Image.open(d/('figure_'+key+'.png'))
        assert abs(im.info['dpi'][0]-300)<1
        for p in d.glob('*.json'):check(json.loads(p.read_text()))
        figures[key]=dict(folder=folder,png_pixels=list(im.size),dpi=im.info['dpi'],
            pdf_sha256=sha(d/('figure_'+key+'.pdf')))
    f05=json.loads((OUT/'F05_case00/sources.json').read_text())
    f06=json.loads((OUT/'F06_five_cases/case_manifest.json').read_text())
    f10=json.loads((OUT/'F10_multiview/sources.json').read_text())
    assert f05['final_scene']['sha256']==f06['cases'][0]['final_scene']['sha256']==f10['scene']['sha256']
    variants=['variant_B_scara','variant_C_cnc','variant_D_conveyor']
    d=OUT/'F09_text_variants'
    for n in ['anchor.png','position_hint.png','depth_background.png','depth.png','edit_mask.png','anchor_camera.json']:
        assert len({sha(d/v/n) for v in variants})==1
    check(json.loads((d/'input_manifest.json').read_text()))
    report=dict(figures=figures,source_hashes_rechecked=checked,
        same_case00_final_scene_f05_f06_f10=True,f09_shared_conditions_byte_identical=True,
        no_model_generation_or_training=True,no_source_ply_written=True,
        rendering_only='F05/F06/F10 actual saved Gaussian PLY; F07 accepted v8 images unchanged.',
        visual_review='PDF-generated previews inspected; original photos/Qwen references retain native resolution.')
    (OUT/'validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SCI Figures v3</title>
<style>body{font:16px system-ui,sans-serif;margin:28px auto;max-width:1400px;padding:0 20px;background:#f6f7f8;color:#17222c}h1{font-size:28px}section{background:white;padding:22px;margin:24px 0;border:1px solid #d8dde3;border-radius:8px}img{display:block;max-width:100%;max-height:1800px;height:auto;margin:18px auto}a{color:#165e9b;margin-right:18px}.caption{line-height:1.65;max-width:1000px}p{line-height:1.6}</style>
<h1>SCI 论文图 · paper_figures_v3</h1>
<p>真实已有实验结果。F05/F06/F10 从保存的 Gaussian PLY 渲染；F07 保留已确认的 v8 视角和 GSLE Coarse 6500。F09 仅准备输入。</p>
<p><a href="paper_figures_v3.zip">下载完整材料 ZIP</a><a href="validation_report.json">来源与校验记录</a><a href="README.txt">使用说明</a></p>
<section><h2>F09 · Qwen 输入</h2><p>每组按 Anchor RGB → Position Hint → Background Depth 的顺序提交三张图，再粘贴同组 generation_prompt.txt。输入条件保持相同，只改变目标文本。</p>'''
    for name,label in [('scara','SCARA'),('cnc','CNC'),('conveyor','滚筒输送机'),('all_variants','全部三个方案')]:
        body+=f'<a href="F09_text_variants/case00_{name}.zip">{label} ZIP</a>'
    body+='<p><a href="F09_text_variants/prompts_summary.txt">完整英文 Prompt</a><a href="F09_text_variants/README.txt">生成结果导入方法</a></p><img src="F09_text_variants/input_overview.png"></section>'
    for key,folder in folders.items():
        title={'F05':'Case00 完整应用展示','F06':'五类设备阶段结果','F07':'定位与替换结果比较','F10':'Case00 五方向多视角'}[key]
        body+=f'<section><h2>{key} · {title}</h2><p><a href="{folder}/figure_{key}.pdf">PDF</a><a href="{folder}/figure_{key}.png">300 DPI PNG</a><a href="{folder}/caption.txt">英文图注</a><a href="{folder}/">独立子图与来源</a></p>'
        body+=f'<img loading="lazy" src="{folder}/figure_{key}_preview.jpg"><p class="caption">{html.escape((OUT/folder/"caption.txt").read_text())}</p></section>'
    body+='</html>'
    (OUT/'index.html').write_text(body,encoding='utf-8')
    (OUT/'README.txt').write_text('SCI figures v3\n\nF05, F06, F07, F10: 300-DPI PNG + PDF + original independent panels + English caption + source/camera records.\n'
        'F07: accepted comparison_updated_v8 cameras and images, GSLE Coarse 6500 (not DI-SDS). Viewpoint-matched, not universally pixel-aligned across methods.\n'
        'F05/F06/F10 Case00: identical saved registration_all_cases_v1/case00/m4/scene_candidate_step2000.ply.\n'
        'F06 Case07: accepted yaw-adjusted final scene. Case02 full Qwen image recovered from its import provenance.\n'
        'F10: true scene occlusions retained. Bottom row is a uniform crop from the same top-row camera; no hidden independent-object substitution.\n'
        'F09: only three Qwen input packages; no new generated outputs. Upload exactly anchor.png, position_hint.png, depth_background.png in that order with generation_prompt.txt.\n'
        'Save each future full-scene Qwen output as edited_scene.png in its variant folder. See F09_text_variants/README.txt for importing into a fresh experiment.\n'
        'No Qwen/TRELLIS/optimization was run and no source PLY was modified.\n'
        'Render scripts: industrial_swap_v2/scripts/paper_figures_v3/. GPU rendering uses envs/gsle/bin/python; PDF assembly uses system python3 + reportlab + pdftoppm.\n',encoding='utf-8')
    scripts=OUT/'reproduction_scripts';scripts.mkdir(exist_ok=True)
    for p in (ROOT/'industrial_swap_v2/scripts/paper_figures_v3').glob('*.py'):shutil.copy2(p,scripts/p.name)
    archive=OUT/'paper_figures_v3.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p!=archive:z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(output=str(OUT),zip_bytes=archive.stat().st_size,checked_sources=len(checked),figures=figures),indent=2),flush=True)


if __name__=='__main__':main()
