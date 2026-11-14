"""Copy the confirmed F11 into the publication collection without editing media."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F11_spatial_conditioning_final_v3'
COLLECTION = ROOT/'experiment figure'
DEST = COLLECTION/'F11'
DATE = '2026-11-12'
DESCRIPTION = ('Case00 spatial conditioning: localized Gaussian highlight, generation space and support; '
               'visibility protection, background depth, and localized RGB analysis.')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert SOURCE.is_dir() and COLLECTION.is_dir()
    assert not DEST.exists(), 'Existing confirmed F11 must not be overwritten.'
    manifest_path = COLLECTION/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    assert 'F11' not in manifest['figures']
    old_main_hashes = {rec[k]: sha(COLLECTION/rec[k]) for rec in manifest['figures'].values()
                       for k in ['png', 'pdf']}
    index_path = COLLECTION/'index.html'
    index = index_path.read_text(encoding='utf-8')
    assert 'F11/index.html' not in index
    assert '<a href="F17/index.html">F17</a>' in index
    assert '<section><h2><a href="F17/index.html">' in index
    records = []
    renamed = {'figure_F11_final.png': 'F11.png', 'figure_F11_final.pdf': 'F11.pdf'}
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file() or src == SOURCE/'index.html':
            continue
        rel = src.relative_to(SOURCE).as_posix()
        dst = DEST/renamed.get(rel, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(src)
        assert sha(dst) == digest
        records.append(dict(file=dst.relative_to(COLLECTION).as_posix(),
                            source=src.relative_to(ROOT).as_posix(), sha256=digest,
                            bytes=dst.stat().st_size))
    (DEST/'README.txt').write_text(
        'F11 - user-confirmed version, '+DATE+'\n'+DESCRIPTION+'\n\n'
        'F11.png: confirmed 300 DPI main figure\n'
        'F11.pdf: confirmed publication PDF with vector annotations\n'
        'panels/: independent high-resolution panels\n'
        'caption.txt: English caption\nsources.json: data, camera and visualization provenance\n'
        'inputs/, analysis/: existing source images, masks, cached Gaussian contribution and figure materials\n'
        'code/: reproducible script snapshots, retaining their original output locations\n'
        'validation.json: panel-A-only revision checks; B-D remain pixel-identical\n'
        'delivery_manifest.json: source mapping and SHA256 copy verification\n\n'
        'Panel A uses semitransparent actual M1 Gaussian contribution, the saved M2 generation box '
        'and support face; no original box, contour, or inset. The confirmed Qwen robot remains '
        'a historical projected-box output, not a direct output of the later adaptive conditions. '
        'All source media were copied without content edits. Previous versions remain intact.\n', encoding='utf-8')
    page = (SOURCE/'index.html').read_text(encoding='utf-8')
    page = page.replace('figure_F11_final.png', 'F11.png').replace('figure_F11_final.pdf', 'F11.pdf')
    page = page.replace('<title>F11 - final A highlight</title>', '<title>F11 - Confirmed</title>')
    (DEST/'index.html').write_text(page, encoding='utf-8')
    write(DEST/'delivery_manifest.json', dict(
        figure='F11', user_confirmed=True, confirmed_on=DATE, description=DESCRIPTION,
        approved_source=SOURCE.relative_to(ROOT).as_posix(), files=records,
        all_copied_files_sha256_verified=True, no_media_content_edits=True,
        generated_archive_files=['README.txt', 'index.html', 'delivery_manifest.json']))
    files = [p for p in DEST.rglob('*') if p.is_file()]
    manifest['figures']['F11'] = dict(description=DESCRIPTION, approved_source=SOURCE.name,
        confirmed_on=DATE, png='F11/F11.png', pdf='F11/F11.pdf', files=len(files),
        bytes=sum(p.stat().st_size for p in files),
        main_png_sha256=sha(DEST/'F11.png'), main_pdf_sha256=sha(DEST/'F11.pdf'))
    manifest['figures'] = dict(sorted(manifest['figures'].items()))
    manifest['last_updated_on'] = DATE
    write(manifest_path, manifest)
    index = index.replace('F05、F06、F07、F08、F10、F17', 'F05、F06、F07、F08、F10、F11、F17')
    index = index.replace('<a href="F17/index.html">F17</a>',
                          '<a href="F11/index.html">F11</a><a href="F17/index.html">F17</a>', 1)
    section = ('<section><h2><a href="F11/index.html">F11</a></h2>'
        '<p>Case00 空间条件：真实高斯半透明高亮、生成空间与支撑面；前景保护、背景深度及局部 RGB 差异。</p>'
        '<a href="F11/F11.png"><img src="F11/F11.png" loading="lazy"></a></section>')
    index = index.replace('<section><h2><a href="F17/index.html">',
                          section+'<section><h2><a href="F17/index.html">', 1)
    index_path.write_text(index, encoding='utf-8')
    readme_path = COLLECTION/'README.txt'
    readme_path.write_text(readme_path.read_text(encoding='utf-8').rstrip()+
                          '\n\nF11 ('+DATE+'): '+DESCRIPTION+'\n', encoding='utf-8')
    assert all(sha(COLLECTION/p) == digest for p, digest in old_main_hashes.items())
    print(json.dumps(dict(directory=str(DEST), verified_copies=len(records), files=len(files),
        other_confirmed_figures_unchanged=True, png_sha256=sha(DEST/'F11.png'),
        pdf_sha256=sha(DEST/'F11.pdf')), ensure_ascii=False))


if __name__ == '__main__':
    main()
