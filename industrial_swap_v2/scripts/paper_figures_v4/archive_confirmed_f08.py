"""Archive the user-confirmed F08 without altering figure or experiment content."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F08_final_M1_06_text_query'
COLLECTION = ROOT / 'experiment figure'
DEST = COLLECTION / 'F08'
CONFIRMED = '2026-11-12'
DESCRIPTION = ('Photo-guided instance localization, M1-06; centered Text query; '
               '10-case selection summary (4/10 and 10/10).')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert SOURCE.is_dir() and COLLECTION.is_dir()
    assert not DEST.exists(), 'F08 already exists; do not overwrite a confirmed archive.'
    manifest_path = COLLECTION / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    assert 'F08' not in manifest['figures']
    records = []
    rename = {'figure_F08_final.png': 'F08.png', 'figure_F08_final.pdf': 'F08.pdf'}
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file() or src == SOURCE / 'index.html':
            continue
        relative = src.relative_to(SOURCE)
        dst = DEST / rename.get(relative.as_posix(), relative.as_posix())
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(src)
        assert sha(dst) == digest
        records.append(dict(file=dst.relative_to(COLLECTION).as_posix(),
                            source=src.relative_to(ROOT).as_posix(), sha256=digest,
                            bytes=dst.stat().st_size))
    (DEST / 'README.txt').write_text(
        'F08 - user-confirmed figure, ' + CONFIRMED + '\n' + DESCRIPTION + '\n\n'
        'F08.png: confirmed 300 DPI image\nF08.pdf: vector-label publication PDF\n'
        'panels/: independent figure panels\ncaption.txt: English figure caption\n'
        'instance_selection.csv: original saved judgments for 10 M1 cases\n'
        'statistics_summary.csv: 4/10 and 10/10 case-wise summary\n'
        'sources.json: source images, cameras, Gaussian renders and statistics provenance\n'
        'inputs/, renders/: existing materials used to compose the figure\n'
        'code/: reproducible figure script snapshot (retains its original output directory)\n'
        'delivery_manifest.json: source mapping and SHA256 checks\n\n'
        'The 10 M1 cases include 9 unique physical targets. Judgments remain a preliminary '
        'AI visual audit. No image, PDF, Gaussian, camera or experiment content was changed. '
        'Earlier versions remain in their original directories.\n', encoding='utf-8')
    page = (SOURCE / 'index.html').read_text(encoding='utf-8')
    page = page.replace('figure_F08_final.png', 'F08.png').replace('figure_F08_final.pdf', 'F08.pdf')
    page = page.replace('<h1>F08</h1>', '<h1>F08 · Confirmed</h1>')
    (DEST / 'index.html').write_text(page, encoding='utf-8')
    write_json(DEST / 'delivery_manifest.json', dict(
        figure='F08', user_confirmed=True, confirmed_on=CONFIRMED, description=DESCRIPTION,
        original_figure_directory=str(SOURCE), files=records, all_copied_files_hash_verified=True,
        no_pdf_or_image_content_edits=True, generated_archive_files=['README.txt', 'index.html']))
    files = [p for p in DEST.rglob('*') if p.is_file()]
    manifest['figures']['F08'] = dict(
        description=DESCRIPTION, approved_source=SOURCE.name, confirmed_on=CONFIRMED,
        png='F08/F08.png', pdf='F08/F08.pdf', files=len(files),
        bytes=sum(p.stat().st_size for p in files),
        main_png_sha256=sha(DEST / 'F08.png'), main_pdf_sha256=sha(DEST / 'F08.pdf'))
    manifest['figures'] = dict(sorted(manifest['figures'].items()))
    manifest['last_updated_on'] = CONFIRMED
    write_json(manifest_path, manifest)
    index_path = COLLECTION / 'index.html'
    index = index_path.read_text(encoding='utf-8')
    assert 'F08/index.html' not in index
    index = index.replace('F05、F06、F07、F10、F17', 'F05、F06、F07、F08、F10、F17')
    nav = '<a href="F10/index.html">F10</a>'
    index = index.replace(nav, '<a href="F08/index.html">F08</a>' + nav, 1)
    section = ('<section><h2><a href="F08/index.html">F08</a></h2>'
               '<p>M1-06 照片引导实例定位；Text query 共享输入；10 个 M1 案例统计：4/10 与 10/10。</p>'
               '<a href="F08/F08.png"><img src="F08/F08.png" loading="lazy"></a></section>')
    index = index.replace('<section><h2><a href="F10/index.html">', section + '<section><h2><a href="F10/index.html">', 1)
    index_path.write_text(index, encoding='utf-8')
    readme = COLLECTION / 'README.txt'
    readme.write_text(readme.read_text(encoding='utf-8').rstrip() + '\n\nF08 (' + CONFIRMED + '): ' + DESCRIPTION + '\n', encoding='utf-8')
    print(json.dumps(dict(directory=str(DEST), files=len(files),
                          copied_files_verified=len(records), png_sha256=sha(DEST / 'F08.png'),
                          pdf_sha256=sha(DEST / 'F08.pdf'))))


if __name__ == '__main__':
    main()
