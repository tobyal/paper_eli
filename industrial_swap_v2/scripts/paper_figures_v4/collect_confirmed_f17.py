"""Add the user-confirmed F17 to experiment figure without changing prior figures."""
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F17_localization_comparison_final'
DEST=ROOT/'experiment figure'
TARGET=DEST/'F17'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    previous={str(p.relative_to(DEST)):sha(p) for folder in DEST.iterdir() if folder.is_dir() and folder.name!='F17'
              for p in folder.rglob('*') if p.is_file()}
    TARGET.mkdir(exist_ok=True)
    records=[]
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file():continue
        relative=src.relative_to(SOURCE)
        if relative==Path('index.html'):continue
        if relative.name=='figure_F17_final.png':relative=Path('F17.png')
        elif relative.name=='figure_F17_final.pdf':relative=Path('F17.pdf')
        elif relative==Path('pdf_proof.png'):relative=Path('print_proof.png')
        dst=TARGET/relative
        digest=sha(src)
        if dst.exists() and sha(dst)!=digest:
            raise RuntimeError('Different confirmed file already exists: '+str(dst))
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(dst)==digest
        records.append(dict(file=str(dst.relative_to(DEST)).replace('\\','/'),
            source=str(src.relative_to(ROOT)).replace('\\','/'),sha256=digest,bytes=src.stat().st_size))
    description='Gaussian instance-localization comparison; square qualitative panels and enlarged mean-IoU panel (b).'
    manifest=dict(figure='F17',user_confirmed=True,confirmed_on='2026-11-11',
        description=description,original_figure_directory=str(SOURCE),files=records,
        main_png_sha256=sha(TARGET/'F17.png'),main_pdf_sha256=sha(TARGET/'F17.pdf'),
        copied_artifacts_unmodified=True,original_files_preserved=True,
        provenance_note='The confirmed user-polygon GT, actual Gaussian projections, metrics and camera records are preserved. Clean delivery filenames map to their originals here.')
    write(TARGET/'delivery_manifest.json',manifest)
    (TARGET/'README.txt').write_text('F17 - user-confirmed figure\n'+description+'\n\n'
        'F17.png: confirmed 300 DPI image\nF17.pdf: vector-label publication PDF\n'
        'panels/: seven square image panels and the mean-IoU chart\n'
        'caption.txt: English caption\niou_summary.csv: confirmed mean IoU values\n'
        'sources.json: experiment, rendering and camera provenance\n'
        'ground_truth_masks/ and inputs/: confirmed GT and supplied annotation\n'
        'renders/: actual render-derived materials used for the panels\n'
        'code/: reproducible figure script snapshot (retains its original output directory)\n'
        'delivery_manifest.json: source mapping and SHA256 checks\n'
        '\nAll figures and data were copied without content changes. Earlier versions remain in their original directories.\n',encoding='utf-8')
    style='<style>body{font:16px/1.6 Arial,sans-serif;max-width:1600px;margin:24px auto;padding:0 20px;background:white;color:#222}img{width:100%;height:auto}a{color:#185b94}</style>'
    (TARGET/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>F17 · Confirmed</title>'+style+
        '<p><a href="../index.html">返回确认版本总览</a></p><h1>F17</h1>'+
        '<p>已确认版本：方形定位及 Gaussian-only 子图，平滑青色 GT 轮廓，以及放大的 (b) 平均 IoU 图。</p>'+
        '<p><a href="F17.png">高清 PNG</a> · <a href="F17.pdf">PDF</a> · <a href="panels/">独立子图</a> · '+
        '<a href="caption.txt">Caption</a> · <a href="iou_summary.csv">IoU CSV</a> · <a href="sources.json">来源</a> · '+
        '<a href="delivery_manifest.json">文件校验</a></p><img src="F17.png">',encoding='utf-8')
    root_manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf-8'))
    root_manifest['figures']['F17']=dict(description=description,approved_source=SOURCE.name,
        png='F17/F17.png',pdf='F17/F17.pdf',files=len(records),bytes=sum(x['bytes'] for x in records),
        main_png_sha256=manifest['main_png_sha256'],main_pdf_sha256=manifest['main_pdf_sha256'])
    write(DEST/'manifest.json',root_manifest)
    page=(DEST/'index.html').read_text(encoding='utf-8')
    if 'href="F17/index.html"' not in page:
        page=page.replace('</nav>','<a href="F17/index.html">F17</a></nav>',1)
        page=page.replace('F05、F06、F07、F10 为已确认的论文图','F05、F06、F07、F10、F17 为已确认的论文图')
        section='<section><h2><a href="F17/index.html">F17</a></h2><p>三维 Gaussian 实例定位比较：方形定性子图与放大的 (b) 平均 IoU 图。</p><a href="F17/F17.png"><img src="F17/F17.png" loading="lazy"></a></section>'
        page=page.replace('</body>',section+'</body>') if '</body>' in page else page+section
        (DEST/'index.html').write_text(page,encoding='utf-8')
    readme=(DEST/'README.txt').read_text(encoding='utf-8')
    if 'F17:' not in readme:
        (DEST/'README.txt').write_text(readme+'\nF17: '+description+'\n',encoding='utf-8')
    assert all(sha(DEST/p)==digest for p,digest in previous.items())
    print(json.dumps(dict(destination=str(TARGET),copied_files=len(records),
        other_confirmed_figures_unchanged=True,PNG_sha256=manifest['main_png_sha256'],
        PDF_sha256=manifest['main_pdf_sha256']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
