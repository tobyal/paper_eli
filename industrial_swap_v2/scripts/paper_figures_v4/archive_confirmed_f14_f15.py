"""Copy the confirmed F14/F15 into the publication collection, byte for byte."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / 'experiment figure'
DATE = '2026-11-13'
FIGURES = {
    'F14': {
        'source': 'F14_occlusion_refinement_insets_flush',
        'stem': 'figure_F14_final',
        'title': 'Residual Gaussian Occlusion Refinement',
        'description': 'Case00 residual Gaussian occlusion: three scene comparisons with base insets flush to the upper-right edges; dual masks, actual opacity adjustment and held-out curves. Full M4 remains a guard-rejected candidate.',
        'summary': 'Case00 残留高斯遮挡优化；底座放大框贴齐右上角，保留双 Mask、透明度调整和真实评价曲线。Full M4 保留 Candidate 标识。',
        'notes': 'Full M4 Candidate was not accepted by the environment guard; the guarded final scene remains Step 0. The archive does not change this experimental status.\n',
    },
    'F15': {
        'source': 'F15_computational_efficiency_final',
        'stem': 'figure_F15_revised',
        'title': 'Computational Cost and Workflow Reusability',
        'description': 'Mean computational cost across five cases and workflow reusability; updated existing-asset reuse labels and no sample-size labels in panel b. Valid-record counts remain in the source data.',
        'summary': '有效运行记录的平均计算成本与流程复用；已更新 Existing Asset Reuse 表述，(b) 不显示 n 标注。',
        'notes': 'The module stack sums recorded operation means, not a measured end-to-end runtime. Missing measurements are excluded. Valid-record counts and runtime provenance remain in the CSV files.\n',
    },
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_text(path, text):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    manifest_path = COLLECTION / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    old_hashes = {record[key]: sha(COLLECTION / record[key])
                  for record in manifest['figures'].values() for key in ('png', 'pdf')}
    index_path = COLLECTION / 'index.html'
    index = index_path.read_text(encoding='utf-8')
    nav_anchor = '<a href="F17/index.html">F17</a>'
    section_anchor = '<section><h2><a href="F17/index.html">'
    assert nav_anchor in index and section_anchor in index
    for figure, spec in FIGURES.items():
        dest = COLLECTION / figure
        source = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4' / spec['source']
        assert source.is_dir() and (source / (spec['stem'] + '.png')).is_file()
        assert not dest.exists() and not dest.is_symlink(), 'Preserve any existing confirmed version.'
        assert dest.resolve().parent == COLLECTION.resolve()
        assert figure not in manifest['figures'] and f'{figure}/index.html' not in index

    results = {}
    for figure, spec in FIGURES.items():
        source = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4' / spec['source']
        dest = COLLECTION / figure
        renames = {spec['stem'] + ext: figure + ext for ext in ('.png', '.pdf')}
        renames['README.txt'] = 'source_README.txt'
        records = []
        for src in sorted(source.rglob('*')):
            if not src.is_file():
                continue
            relative = src.relative_to(source).as_posix()
            assert relative not in ('index.html', 'delivery_manifest.json')
            dst = dest / renames.get(relative, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            digest = sha(src)
            assert sha(dst) == digest
            records.append({'file': dst.relative_to(COLLECTION).as_posix(),
                            'source': src.relative_to(ROOT).as_posix(),
                            'sha256': digest, 'bytes': dst.stat().st_size})
        write_text(dest / 'README.txt',
                   f'{figure} - user-confirmed version, {DATE}\n{spec["description"]}\n\n'
                   f'{figure}.png / {figure}.pdf: confirmed publication figure, no version suffix.\n'
                   'panels/: original independent figure panels.\n'
                   'caption.txt / sources.json: unchanged caption and provenance.\n'
                   'code/: original plotting script snapshot.\n'
                   'All associated data, records and previews are retained.\n'
                   'delivery_manifest.json: original source mapping and verified SHA256 hashes.\n\n'
                   + spec['notes'] + 'No media, experimental results or other confirmed figures were modified.\n')
        write_text(dest / 'index.html',
                   f'<!doctype html><meta charset="utf-8"><title>{figure} - Confirmed</title>'
                   '<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222}'
                   'img{width:100%;height:auto}a{color:#185b94}nav a{margin-right:20px}</style>'
                   '<nav><a href="../index.html">All confirmed figures</a>'
                   f'<a href="{figure}.png">PNG</a><a href="{figure}.pdf">PDF</a>'
                   '<a href="caption.txt">Caption</a><a href="sources.json">Sources</a>'
                   '<a href="delivery_manifest.json">Archive manifest</a></nav>'
                   f'<h1>{figure} - {spec["title"]}</h1>'
                   f'<a href="{figure}.png"><img src="pdf_preview.png" alt="{figure}"></a>')
        write_json(dest / 'delivery_manifest.json', {
            'figure': figure, 'user_confirmed': True, 'confirmed_on': DATE,
            'description': spec['description'], 'approved_source': source.relative_to(ROOT).as_posix(),
            'files': records, 'all_copied_files_sha256_verified': True,
            'no_media_content_edits': True,
            'generated_archive_files': ['README.txt', 'index.html', 'delivery_manifest.json'],
        })
        files = [path for path in dest.rglob('*') if path.is_file()]
        manifest['figures'][figure] = {
            'description': spec['description'], 'approved_source': source.name, 'confirmed_on': DATE,
            'png': f'{figure}/{figure}.png', 'pdf': f'{figure}/{figure}.pdf', 'files': len(files),
            'bytes': sum(path.stat().st_size for path in files),
            'main_png_sha256': sha(dest / f'{figure}.png'),
            'main_pdf_sha256': sha(dest / f'{figure}.pdf'),
        }
        results[figure] = {'verified_copies': len(records), 'files': len(files),
                           'png_sha256': sha(dest / f'{figure}.png'),
                           'pdf_sha256': sha(dest / f'{figure}.pdf')}
        index = index.replace(nav_anchor, f'<a href="{figure}/index.html">{figure}</a>' + nav_anchor, 1)
        section = (f'<section><h2><a href="{figure}/index.html">{figure}</a></h2>'
                   f'<p>{spec["summary"]}</p><a href="{figure}/{figure}.png">'
                   f'<img src="{figure}/pdf_preview.png" loading="lazy"></a></section>')
        index = index.replace(section_anchor, section + section_anchor, 1)

    manifest['figures'] = dict(sorted(manifest['figures'].items()))
    manifest['last_updated_on'] = DATE
    write_json(manifest_path, manifest)
    index = index.replace('F12、F13、F17', 'F12、F13、F14、F15、F17')
    write_text(index_path, index)
    readme = COLLECTION / 'README.txt'
    text = readme.read_text(encoding='utf-8').rstrip()
    for figure, spec in FIGURES.items():
        text += f'\n\n{figure} ({DATE}): {spec["description"]}'
    write_text(readme, text + '\n')
    assert all(sha(COLLECTION / name) == digest for name, digest in old_hashes.items())
    for page in [index_path] + [COLLECTION / figure / 'index.html' for figure in FIGURES]:
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
            assert (page.parent / link).is_file(), (page, link)
    print(json.dumps({'archived': results, 'other_confirmed_figures_unchanged': True,
                      'index_links_verified': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
