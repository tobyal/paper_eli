"""Prepare and register the real F12, replacing the collection's illustration.

Windows replacement is performed with checked, native PowerShell file operations.
No experiment or image/PDF content is modified by this archival operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F12_multicase_training_final'
COLLECTION = ROOT/'experiment figure'
STAGE = COLLECTION/'.F12_empirical_staging'
DEST = COLLECTION/'F12'
DATE = '2026-11-12'
EMPIRICAL_PNG = '594ebeb551efc65e3358627309c36ea28d8f852a8807346a18261b3aab0f0ea0'
ILLUSTRATION_PNG = '4d2e2899c82f752c83bceea5ab86bfeb022c35cfbb3140363aa26ec71b4b0e9c'
DESCRIPTION = 'Measured multi-view RGB-D Gaussian reconstruction across five industrial devices; original Training and Held-out series.'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def prepare():
    assert not STAGE.exists(), 'Staging directory already exists; inspect before continuing.'
    assert sha(SOURCE/'figure_F12_final.png') == EMPIRICAL_PNG
    provenance = json.loads((SOURCE/'sources.json').read_text(encoding='utf-8'))
    assert len(provenance['cases']) == 5 and provenance['last_common_step'] == 4000
    assert not provenance.get('hypothetical', False)
    if DEST.exists():
        assert DEST.is_dir() and not DEST.is_symlink()
        assert sha(DEST/'F12.png') == ILLUSTRATION_PNG, 'Existing F12 is not the reviewed illustration.'
    records = []
    names = {'figure_F12_final.png':'F12.png', 'figure_F12_final.pdf':'F12.pdf',
             'preview_F12_final.jpg':'preview.jpg'}
    for src in sorted(SOURCE.rglob('*')):
        if src.is_file():
            rel = src.relative_to(SOURCE).as_posix()
            dst = STAGE/names.get(rel, rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
            records.append(dict(file=dst.relative_to(STAGE).as_posix(),
                                source=src.relative_to(ROOT).as_posix(), sha256=sha(src)))
    for name in ['figure_f12_multicase_final.py', 'figure_common.py']:
        src = ROOT/'industrial_swap_v2/scripts/paper_figures_v4'/name
        (STAGE/'code').mkdir(exist_ok=True)
        shutil.copy2(src, STAGE/'code'/name)
        records.append(dict(file='code/'+name, source=src.relative_to(ROOT).as_posix(), sha256=sha(src)))
    write(STAGE/'delivery_manifest.json', dict(
        figure='F12', artifact_type='empirical_experimental_result', confirmed_on=DATE,
        approved_source=SOURCE.relative_to(ROOT).as_posix(), files=records,
        original_data_preserved=True, all_copies_hash_verified=True,
        no_media_content_edits=True, no_new_training=True))
    (STAGE/'README.txt').write_text(
        'F12 - verified experimental results, '+DATE+'\n'+DESCRIPTION+'\n\n'
        'F12.png / F12.pdf: unchanged measured-result figure\n'
        'panels/: four original panels\nmulticase_convergence.csv: actual per-case and aggregate metrics\n'
        'caption.txt: full metric definitions and evaluation protocol\n'
        'sources.json: original log paths, hashes, selected checkpoints and numerical summaries\n'
        'code/: plotting source snapshots; original project paths are retained\n'
        'delivery_manifest.json: byte-preserving source mapping\n\n'
        'Training and held-out depth curves retain their original identities. The held-out SD band '
        'is computed from the five actual cases. The training/validation view counts are 30/6 per device. '
        'The user requested replacing the illustrative collection version with these measured results.\n', encoding='utf-8')
    (STAGE/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>F12 - Experimental Results</title>'
        '<style>body{font:16px/1.5 Arial;max-width:1500px;margin:24px auto}img{max-width:100%}</style>'
        '<p>F12 · 五种工业设备的真实 RGB-D Gaussian 重建结果</p>'
        '<p><a href="../index.html">Figures</a> | <a href="F12.png">PNG</a> | <a href="F12.pdf">PDF</a> | '
        '<a href="panels/">Panels</a> | <a href="multicase_convergence.csv">Measured data</a> | '
        '<a href="caption.txt">Caption</a> | <a href="sources.json">Sources</a></p><img src="F12.png">', encoding='utf-8')
    print(json.dumps(dict(stage=str(STAGE.resolve()), verified_files=len(records), png=EMPIRICAL_PNG)), flush=True)


def finalize():
    assert DEST.is_dir() and not STAGE.exists()
    delivery = json.loads((DEST/'delivery_manifest.json').read_text(encoding='utf-8'))
    assert sha(DEST/'F12.png') == EMPIRICAL_PNG
    assert not (DEST/'illustrative_depth_curves.csv').exists()
    assert all(sha(DEST/r['file']) == r['sha256'] for r in delivery['files'])
    mp = COLLECTION/'manifest.json'
    manifest = json.loads(mp.read_text(encoding='utf-8-sig'))
    other_hashes = {r[k]:sha(COLLECTION/r[k]) for key,r in manifest['figures'].items()
                    if key != 'F12' for k in ['png','pdf']}
    files = [p for p in DEST.rglob('*') if p.is_file()]
    manifest['figures']['F12'] = dict(description=DESCRIPTION,
        approved_source=SOURCE.relative_to(ROOT).as_posix(), artifact_type='empirical_experimental_result',
        confirmed_on=DATE, png='F12/F12.png', pdf='F12/F12.pdf', files=len(files),
        bytes=sum(p.stat().st_size for p in files), main_png_sha256=sha(DEST/'F12.png'),
        main_pdf_sha256=sha(DEST/'F12.pdf'))
    manifest['figures'] = dict(sorted(manifest['figures'].items()))
    manifest['last_updated_on'] = DATE
    write(mp, manifest)
    index_path = COLLECTION/'index.html'
    index = index_path.read_text(encoding='utf-8')
    section = ('<section><h2><a href="F12/index.html">F12</a></h2>'
        '<p>五种工业设备的真实 RGB-D Gaussian 重建结果：Training / Held-out 收敛及优化前后指标。</p>'
        '<a href="F12/F12.png"><img src="F12/preview.jpg" loading="lazy"></a></section>')
    pattern = r'<section><h2><a href="F12/index.html">.*?</section>'
    if re.search(pattern, index, flags=re.S):
        index = re.sub(pattern, section, index, count=1, flags=re.S)
    else:
        index = index.replace('<section><h2><a href="F17/index.html">',
                              section+'<section><h2><a href="F17/index.html">', 1)
        index = index.replace('<a href="F17/index.html">F17</a>',
                              '<a href="F12/index.html">F12</a><a href="F17/index.html">F17</a>', 1)
        index = index.replace('F10、F11、F17', 'F10、F11、F12、F17')
    index_path.write_text(index, encoding='utf-8')
    readme = COLLECTION/'README.txt'
    content = readme.read_text(encoding='utf-8')
    content = re.sub(r'^F12[^\n]*\n?', '', content, flags=re.M).rstrip()
    readme.write_text(content+'\n\nF12 ('+DATE+'): '+DESCRIPTION+'\n', encoding='utf-8')
    assert all(sha(COLLECTION/p) == h for p,h in other_hashes.items())
    print(json.dumps(dict(directory=str(DEST), files=len(files), real_result_verified=True,
        no_illustrative_csv=True, other_figures_unchanged=True,
        png=sha(DEST/'F12.png'), pdf=sha(DEST/'F12.pdf'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare','finalize'], required=True)
    args = p.parse_args(); prepare() if args.stage == 'prepare' else finalize()
