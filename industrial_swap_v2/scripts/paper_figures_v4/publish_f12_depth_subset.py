"""Publish the requested panel-b subset analysis, preserving original measurements."""
import json
from pathlib import Path
import re
import shutil

from archive_confirmed_f14_f15 import sha, write_json, write_text

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / 'experiment figure'
SOURCE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F12_depth_four_case_summary'
DATE = '2026-11-13'


def main():
    dest = COLLECTION / 'F12'
    staging = COLLECTION / '.F12_depth_subset_staging'
    history = COLLECTION / '_history/F12_20261113_before_depth_subset'
    for path in (dest, staging, history):
        assert not path.is_symlink()
        path.resolve().relative_to(COLLECTION.resolve())
    assert dest.resolve().parent == staging.resolve().parent == COLLECTION.resolve()
    assert history.resolve().parent == COLLECTION.resolve() / '_history'
    assert dest.is_dir() and not staging.exists() and not history.exists()
    assert sha(dest / 'sources.json') == sha(SOURCE / 'inputs/previous_F12_sources.json')
    meta = json.loads((SOURCE / 'sources.json').read_text(encoding='utf-8'))
    assert meta['panel_b']['n_cases'] == 4 and meta['panel_b']['invalid_data_evidence'] is False
    for name in ('a', 'c', 'd'):
        assert sha(dest / 'panels' / (name + '.png')) == sha(SOURCE / 'panels' / (name + '.png'))
    manifest = json.loads((COLLECTION / 'manifest.json').read_text(encoding='utf-8-sig'))
    other = {r[k]: sha(COLLECTION / r[k]) for key, r in manifest['figures'].items()
             if key != 'F12' for k in ('png', 'pdf')}
    prior = {p.relative_to(dest).as_posix(): sha(p) for p in dest.rglob('*') if p.is_file()}
    renames = {'figure_F12_revised.png': 'F12.png', 'figure_F12_revised.pdf': 'F12.pdf',
               'README.txt': 'source_README.txt'}
    files = []
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE).as_posix()
        target = staging / renames.get(rel, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        assert sha(src) == sha(target)
        files.append({'file': 'F12/' + target.relative_to(staging).as_posix(),
                      'source': src.relative_to(ROOT).as_posix(), 'sha256': sha(target),
                      'bytes': target.stat().st_size})
    description = ('Only panel b changed: post-hoc leave-Case04-out depth sensitivity summary '
                   '(Case00/02/07/10, n=4). Panels a/c/d and all raw five-case observations retained.')
    write_text(staging / 'README.txt', 'F12 - user-requested panel-b revision, ' + DATE + '\n' + description +
               '\n\nF12.png / F12.pdf: current figure.\n'
               'depth_convergence_four_cases.csv: panel-b mean and sample SD at every actual step.\n'
               'data/five_case_convergence_original.csv: untouched complete prior records.\n'
               'caption.txt / sources.json: explicit subset scope and reason.\n'
               'Case04 has no detected measurement/aggregation error; this is a sensitivity analysis, not invalid-data removal.\n'
               'Previous figure: ../_history/F12_20261113_before_depth_subset/.\n')
    write_text(staging / 'index.html',
               '<!doctype html><meta charset="utf-8"><title>F12</title>'
               '<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px}img{width:100%;height:auto}a{color:#185b94}nav a{margin-right:20px}</style>'
               '<nav><a href="../index.html">All figures</a><a href="F12.png">PNG</a><a href="F12.pdf">PDF</a>'
               '<a href="caption.txt">Caption</a><a href="depth_convergence_four_cases.csv">Panel-b data</a>'
               '<a href="sources.json">Sources</a><a href="../_history/F12_20261113_before_depth_subset/F12.png">Previous version</a></nav>'
               '<h1>F12</h1><a href="F12.png"><img src="preview.jpg"></a>')
    write_json(staging / 'delivery_manifest.json', {
        'figure': 'F12', 'updated_on': DATE, 'revision_requested': True, 'user_confirmed': False,
        'current_source': SOURCE.relative_to(ROOT).as_posix(), 'description': description,
        'previous_version': history.relative_to(COLLECTION).as_posix(),
        'files': files, 'all_copied_files_sha256_verified': True,
        'generated_archive_files': ['README.txt', 'index.html', 'delivery_manifest.json']})
    history.parent.mkdir(exist_ok=True)
    dest.rename(history)
    staging.rename(dest)
    assert all(sha(history / p) == digest for p, digest in prior.items())
    for name in ('manifest.json', 'index.html', 'README.txt'):
        shutil.copy2(COLLECTION / name, history / ('collection_' + name + '.before'))
    content_files = [p for p in dest.rglob('*') if p.is_file()]
    manifest['figures']['F12'] = {
        'description': description, 'current_source': SOURCE.name, 'updated_on': DATE,
        'status': 'updated_per_user_request', 'previous_version': history.relative_to(COLLECTION).as_posix(),
        'png': 'F12/F12.png', 'pdf': 'F12/F12.pdf', 'files': len(content_files),
        'bytes': sum(p.stat().st_size for p in content_files),
        'main_png_sha256': sha(dest / 'F12.png'), 'main_pdf_sha256': sha(dest / 'F12.pdf')}
    manifest['last_updated_on'] = DATE
    write_json(COLLECTION / 'manifest.json', manifest)
    index = (COLLECTION / 'index.html').read_text(encoding='utf-8')
    section = ('<section><h2><a href="F12/index.html">F12</a></h2>'
               '<p>RGB-D Gaussian 重建；(b) 为排除 Case04 后的四案例深度敏感性汇总，(a)(c)(d) 不变。</p>'
               '<a href="F12/F12.png"><img src="F12/preview.jpg" loading="lazy"></a></section>')
    index, n = re.subn(r'<section><h2><a href="F12/index.html">.*?</section>', section, index, count=1, flags=re.S)
    assert n == 1
    write_text(COLLECTION / 'index.html', index)
    readme = (COLLECTION / 'README.txt').read_text(encoding='utf-8').rstrip()
    write_text(COLLECTION / 'README.txt', readme + '\n\nF12 update (' + DATE + '): ' + description + '\n')
    assert all(sha(COLLECTION / p) == digest for p, digest in other.items())
    for page in (COLLECTION / 'index.html', dest / 'index.html'):
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
            assert (page.parent / link).is_file()
    print(json.dumps({'current': str(dest), 'backup': str(history), 'other_figures_unchanged': True,
                      'unchanged_panels': ['a', 'c', 'd'], 'png_sha256': sha(dest / 'F12.png'),
                      'pdf_sha256': sha(dest / 'F12.pdf')}))


if __name__ == '__main__':
    main()
