"""Copy the user-confirmed publication versions into experiment figure/Fxx.

No figure is regenerated or edited. Destination collisions with different bytes
are rejected; all copied files are verified against their original SHA256.
"""
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
V4=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
DEST=ROOT/'experiment figure'
SCRIPTS=ROOT/'industrial_swap_v2/scripts/paper_figures_v4'
SPECS={
    'F05':dict(folder='F05_landscape',stem='figure_F05_landscape',
        description='Reference-image-assisted replacement; landscape two-row layout.',
        zh='注册照片与文本 → 定位与二维参考 → 三维场景替换；横向两排版。',
        code=['figure_f05_landscape.py','figure_f05_revised.py']),
    'F06':dict(folder='F06_2d_to_3d',stem='figure_F06_revised',
        description='2D-to-3D equipment generation: Case02, Case04, Case07.',
        zh='Case02 输送机、Case04 冷水机、Case07 存储柜；二维参考与三维资产对照。',
        code=['f06_2d_to_3d.py','figure_f05_revised.py']),
    'F07':dict(folder='F07_with_prompts',stem='figure_F07_with_prompts',
        description='Five-case, four-method comparison with shared italic text instructions.',
        zh='五案例、四方法比较；每行共享英文指令，定位/结果按 35:65 排列。',
        code=['f07_with_prompts.py','relayout_f06_f07.py','figure_f05_revised.py']),
    'F10':dict(folder='F10_selected_views',stem='figure_F10_revised',
        description='One row of five user-selected multiview images.',
        zh='一行五视角：Front 细节、Left Oblique 全景、Right-side 全景、Rear 细节、Elevated 细节。',
        code=['f10_selected_views.py','relayout_f06_f07.py','figure_f05_revised.py']),
}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save_json(p,data):
    Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def mapped_name(rel,fig,spec):
    if len(rel.parts)>1:return rel
    name=rel.name
    if name=='index.html':return None
    if name.startswith(spec['stem']):
        suffix=name[len(spec['stem']):]
        if suffix in ['.png','.pdf']:return Path(fig+suffix)
        if suffix=='_preview.jpg':return Path('preview.jpg')
        if suffix=='_180mm_proof.png':return Path('print_proof.png')
    if fig=='F06' and name=='figure_F06_180mm_proof.png':return Path('print_proof.png')
    if fig=='F07':
        for part in ['A','B']:
            prefix='figure_F07_part'+part
            if name.startswith(prefix):
                suffix=name[len(prefix):]
                if suffix=='_180mm_proof.png':suffix='_print_proof.png'
                return Path('alternatives')/('F07'+part+suffix)
            if name=='caption_part'+part+'.txt':
                return Path('alternatives')/('caption_F07'+part+'.txt')
    return rel


def copy_verified(src,dst):
    expected=sha(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(dst)!=expected:raise RuntimeError('Different destination already exists: '+str(dst))
    else:shutil.copy2(src,dst)
    assert sha(dst)==expected
    return dict(file=str(dst.relative_to(DEST)).replace('\\','/'),
                source=str(src.relative_to(ROOT)).replace('\\','/'),
                sha256=expected,bytes=src.stat().st_size)


STYLE='<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222;background:white}img{width:100%;height:auto}a{color:#185b94}p{line-height:1.6}section{margin:32px 0}nav a{margin-right:22px}</style>'


def main():
    DEST.mkdir(exist_ok=True)
    summary={}
    for fig,spec in SPECS.items():
        src=V4/spec['folder']
        assert (src/(spec['stem']+'.pdf')).is_file()
        target=DEST/fig;target.mkdir(exist_ok=True)
        entries=[]
        for p in sorted(src.rglob('*')):
            if not p.is_file():continue
            rel=mapped_name(p.relative_to(src),fig,spec)
            if rel is not None:entries.append(copy_verified(p,target/rel))
        for script in spec['code']:
            entries.append(copy_verified(SCRIPTS/script,target/'code'/script))
        main_png=target/(fig+'.png');main_pdf=target/(fig+'.pdf')
        assert main_png.is_file() and main_pdf.is_file()
        manifest=dict(figure=fig,user_confirmed=True,confirmed_on='2026-11-11',
            description=spec['description'],original_figure_directory=str(src),
            files=entries,main_png_sha256=sha(main_png),main_pdf_sha256=sha(main_pdf),
            copied_artifacts_unmodified=True,original_files_preserved=True,
            provenance_note='sources.json and validation.json preserve original experiment paths and hashes. This manifest maps the clean delivery names to those originals.')
        save_json(target/'delivery_manifest.json',manifest)
        optional=''
        if fig=='F07':
            optional='<p>备选分页：<a href="alternatives/F07A.png">Part A PNG</a> · <a href="alternatives/F07A.pdf">Part A PDF</a> · <a href="alternatives/F07B.png">Part B PNG</a> · <a href="alternatives/F07B.pdf">Part B PDF</a></p><p><a href="prompts.txt">共享指令与完整 A/B 文本</a></p>'
        code_note=('Code files are snapshots of the corresponding figure scripts and their local helpers. '
                   'Use the recorded server project and original environments for reproduction. '
                   'These scripts retain their original output destinations under industrial_swap_v2/outputs; '
                   'running them does not update this confirmed collection automatically.\n')
        (target/'README.txt').write_text(fig+' - user-confirmed figure\n'+spec['description']+'\n\n'
            'Main files: '+fig+'.png (300 DPI), '+fig+'.pdf (vector labels)\n'
            'panels/: independent high-resolution panels\ncaption.txt: English caption\n'
            'sources.json: original source/camera records\ndelivery_manifest.json: copied-file mapping and SHA256\n'
            'code/: relevant reproducible scripts and helper snapshots\n'
            +('alternatives/: Part A/B pagination alternatives\nprompts.txt: display and full A/B texts\n' if fig=='F07' else '')
            +'\n'+code_note,encoding='utf-8')
        page='<!doctype html><meta charset="utf-8"><title>'+fig+' · Confirmed</title>'+STYLE
        page+='<p><a href="../index.html">返回确认版本总览</a></p><h1>'+fig+'</h1><p>'+spec['zh']+'</p>'
        page+=f'<p><a href="{fig}.png">高清 PNG</a> · <a href="{fig}.pdf">PDF</a> · <a href="panels/">独立子图</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">来源</a> · <a href="delivery_manifest.json">文件校验</a></p>'
        page+=optional+'<img src="preview.jpg">'
        (target/'index.html').write_text(page,encoding='utf-8')
        summary[fig]=dict(description=spec['description'],approved_source=spec['folder'],
            png=fig+'/'+fig+'.png',pdf=fig+'/'+fig+'.pdf',files=len(entries),
            bytes=sum(x['bytes'] for x in entries),
            main_png_sha256=manifest['main_png_sha256'],main_pdf_sha256=manifest['main_pdf_sha256'])
    save_json(DEST/'manifest.json',dict(user_confirmed_on='2026-11-11',figures=summary,
        original_outputs_untouched=True,no_pdf_or_image_content_edits=True))
    (DEST/'README.txt').write_text('User-confirmed experimental figures - 2026-11-11\n\n'
        +'\n'.join(fig+': '+spec['description'] for fig,spec in SPECS.items())
        +'\n\nEach Fxx folder contains Fxx.png, Fxx.pdf, panels, caption, source records, and code. '
        'The F07 alternatives are in F07/alternatives. Prior versions remain in their original directories. '
        'No images, PDFs, Gaussian models, camera parameters, or experiments were regenerated or modified. '
        'Large original Gaussian/model assets are referenced in sources.json rather than duplicated.\n',encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>Experiment Figure · Confirmed</title>'+STYLE
    page+='<h1>Experiment Figure · 确认版本</h1><p>F05、F06、F07、F10 为已确认的论文图。主图文件名统一为 Fxx.png / Fxx.pdf，旧版本保留。</p><nav>'
    page+=''.join(f'<a href="{fig}/index.html">{fig}</a>' for fig in SPECS)+'</nav>'
    for fig,spec in SPECS.items():
        page+=f'<section><h2><a href="{fig}/index.html">{fig}</a></h2><p>{html.escape(spec["zh"])}</p><a href="{fig}/{fig}.png"><img src="{fig}/preview.jpg" loading="lazy"></a></section>'
    (DEST/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),figures=summary),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
