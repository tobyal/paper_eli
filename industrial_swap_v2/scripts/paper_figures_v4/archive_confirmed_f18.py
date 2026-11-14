"""Archive the approved acquisition overview without changing image/PDF bytes."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
figures = root / 'experiment figure'
src = figures / 'Experimental_Setup'
ids = [int(p.name[1:]) for p in figures.iterdir() if p.is_dir() and re.fullmatch(r'F\d+', p.name)]
assert max(ids) == 17, ids
name = f'F{max(ids)+1:02d}'
dest = figures / name
assert not dest.exists(), dest
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

sources = load(src/'sources.json')
assert sources['acquisition_sample_ids'] == [20,92,198,262]
registry_backup = figures / '_history/20261114_before_F18_archive'
registry_backup.mkdir(parents=True, exist_ok=False)
for f in ('manifest.json', 'README.txt', 'index.html'):
    shutil.copy2(figures/f, registry_backup/f)

shutil.copytree(src, dest, ignore=shutil.ignore_patterns('_history', '__pycache__', '*.pyc'))
for ext in ('png','pdf','svg'):
    old = dest/f'figure_industrial_scene_overview.{ext}'
    old.rename(dest/f'{name}.{ext}')
    assert sha(dest/f'{name}.{ext}') == sha(src/f'figure_industrial_scene_overview.{ext}')
(dest/'caption.txt').write_text((src/'caption.txt').read_text(encoding='utf-8').replace('Fig. X.', 'Fig. 18.', 1), encoding='utf-8')
sources['confirmed_archive'] = {'figure_id': name, 'confirmed_on': '2026-11-14',
    'approved_source': 'experiment figure/Experimental_Setup',
    'acquisition_sample_ids':[20,92,198,262], 'figure_content_unchanged':True,
    'history_retained_at':'experiment figure/Experimental_Setup/_history'}
save(dest/'sources.json', sources)
(dest/'README.txt').write_text(
    'F18 - User-confirmed industrial scene acquisition overview (2026-11-14).\n'
    'F18.png / F18.pdf / F18.svg contain the approved figure, unchanged.\n'
    'Samples: 020, 092, 198, 262; original 3DGS at registered sample 213.\n'
    'panels/: independent panels. data/: real camera/sample data and review provenance.\n'
    'code/: source scripts, retained for provenance; their original execution location is\n'
    'industrial_swap_v2/scripts/paper_figures_v4/.\n'
    'camera_manifest.json and sources.json retain camera and experiment provenance.\n'
    'The unchanged data summary table is included in PNG/PDF/SVG/Markdown/LaTeX.\n'
    'Earlier layouts remain in ../Experimental_Setup/_history/.\n', encoding='utf-8')
(dest/'index.html').write_text(
    '<!doctype html><meta charset="utf-8"><title>F18</title>'
    '<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}</style>'
    '<p><a href="../index.html">All confirmed figures</a></p><h1>F18</h1>'
    '<p>Industrial video acquisition, COLMAP trajectory and original 3DGS reconstruction.</p>'
    '<p><a href="F18.png">PNG</a> | <a href="F18.pdf">PDF</a> | <a href="F18.svg">SVG</a> | <a href="caption.txt">Caption</a></p>'
    '<img src="F18.png"><p><a href="table_industrial_scene_summary.pdf">Data summary table</a> | <a href="sources.json">Sources</a></p>',encoding='utf-8')
files = [p for p in dest.rglob('*') if p.is_file()]
manifest = load(figures/'manifest.json')
manifest['figures'][name] = {
    'description':'Industrial video acquisition (samples 020, 092, 198, 262), true COLMAP trajectory and original 3DGS reconstruction.',
    'approved_source':'Experimental_Setup', 'confirmed_on':'2026-11-14',
    'png':'F18/F18.png','pdf':'F18/F18.pdf','svg':'F18/F18.svg',
    'files':len(files),'bytes':sum(p.stat().st_size for p in files),
    'main_png_sha256':sha(dest/'F18.png'),'main_pdf_sha256':sha(dest/'F18.pdf')}
manifest['last_updated_on']='2026-11-14'
save(figures/'manifest.json',manifest)
readme=figures/'README.txt'
readme.write_text(readme.read_text(encoding='utf-8')+'\nF18 (2026-11-14): Confirmed industrial scene acquisition overview; samples 020/092/198/262, real COLMAP camera trajectory and original 3DGS. Includes data table and provenance.\n',encoding='utf-8')
index=figures/'index.html'
html=index.read_text(encoding='utf-8')
html=html.replace('F16、F17 为已确认', 'F16、F17、F18 为已确认')
html=html.replace('</nav>','<a href="F18/index.html">F18</a></nav>',1)
html += '<section><h2><a href="F18/index.html">F18</a></h2><p>真实工业场景采集总览：020、092、198、262 采样帧，COLMAP 相机轨迹及原始 3DGS 重建。</p><a href="F18/F18.png"><img src="F18/F18.png" loading="lazy"></a></section>\n'
index.write_text(html, encoding='utf-8')
for ext in ('png','pdf','svg','md','tex'):
    f=f'table_industrial_scene_summary.{ext}'
    assert sha(dest/f)==sha(src/f)
print(json.dumps({'archive':str(dest),'figure_id':name,'figure_unchanged':True,'files':len(files),'png_sha256':sha(dest/'F18.png'),'pdf_sha256':sha(dest/'F18.pdf')},ensure_ascii=False))
