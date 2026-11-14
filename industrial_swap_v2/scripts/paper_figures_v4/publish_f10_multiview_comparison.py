"""Update current F10 after visual review, preserving its complete prior version."""
import json
from pathlib import Path
import re
import shutil
import sys

from archive_confirmed_f14_f15 import sha, write_json, write_text

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / 'experiment figure'
DUALMASK = '--dualmask' in sys.argv
SOURCE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4' / ('F10_multiview_comparison_dualmask' if DUALMASK else 'F10_multiview_comparison')
DATE = '2026-11-13'


def main():
    dest = COLLECTION / 'F10'
    staging = COLLECTION / '.F10_multiview_staging'
    history_name = 'F10_20261113_before_dualmask_correction' if DUALMASK else 'F10_20261113_before_multiview_comparison'
    history = COLLECTION / '_history' / history_name
    for path in (dest, staging, history):
        assert not path.is_symlink()
        path.resolve().relative_to(COLLECTION.resolve())
    assert dest.resolve().parent == staging.resolve().parent == COLLECTION.resolve()
    assert history.resolve().parent == COLLECTION.resolve() / '_history'
    assert dest.is_dir() and not staging.exists() and not history.exists()
    assert sha(dest / 'sources.json') == sha(SOURCE / 'inputs/previous_F10_sources.json')
    manifest = json.loads((COLLECTION / 'manifest.json').read_text(encoding='utf-8-sig'))
    other_hashes = {r[k]: sha(COLLECTION / r[k]) for key, r in manifest['figures'].items()
                    if key != 'F10' for k in ('png', 'pdf')}
    prior_files = {p.relative_to(dest).as_posix(): sha(p) for p in dest.rglob('*') if p.is_file()}
    rename = {'figure_F10_revised.png': 'F10.png', 'figure_F10_revised.pdf': 'F10.pdf',
              'README.txt': 'source_README.txt'}
    records = []
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE).as_posix()
        assert rel not in ('delivery_manifest.json', 'index.html')
        output = staging / rename.get(rel, rel)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output)
        assert sha(src) == sha(output)
        records.append({'file': 'F10/' + output.relative_to(staging).as_posix(),
                        'source': src.relative_to(ROOT).as_posix(),
                        'sha256': sha(output), 'bytes': output.stat().st_size})
    description = ('Case00 original/reconfigured 3DGS, 2 rows x 5 matched-camera views. '
                   + ('Confirmed camera poses/crops preserved; restored user-selected dual-mask optimized Step-2000 scene (candidate status retained).' if DUALMASK else
                      'Confirmed camera poses/crops preserved; stored guarded scene_final (Step 0 background).'))
    write_text(staging / 'README.txt',
               'F10 - current user-requested revision, ' + DATE + '\n' + description + '\n\n'
               'F10.png / F10.pdf: current publication figure.\n'
               'panels/: ten equal-size before/after panels.\n'
               'camera_manifest.json: exact camera pairing and crop parameters.\n'
               'renders/: full-resolution Gaussian renders.\n'
               'caption.txt / sources.json: scene and guard provenance.\n'
               'inputs/: previous confirmed source records.\n'
               'Previous F10: ../_history/' + history_name + '/\n')
    write_text(staging / 'index.html',
               '<!doctype html><meta charset="utf-8"><title>F10 - Multi-view comparison</title>'
               '<style>body{font-family:Arial,sans-serif;max-width:1700px;margin:24px auto;padding:0 20px;color:#222}'
               'img{width:100%;height:auto}a{color:#185b94}nav a{margin-right:20px}</style>'
               '<nav><a href="../index.html">All figures</a><a href="F10.png">PNG</a>'
               '<a href="F10.pdf">PDF</a><a href="caption.txt">Caption</a>'
               '<a href="camera_manifest.json">Cameras</a><a href="sources.json">Sources</a>'
               '<a href="../_history/' + history_name + '/F10.png">Previous F10</a></nav>'
               '<h1>F10 - Original / Reconfigured 3DGS</h1><a href="F10.png"><img src="preview.jpg"></a>')
    write_json(staging / 'delivery_manifest.json', {
        'figure': 'F10', 'updated_on': DATE, 'revision_requested': True,
        'based_on_user_confirmed_version': True, 'user_confirmed': False,
        'description': description, 'current_source': SOURCE.relative_to(ROOT).as_posix(),
        'previous_version': history.relative_to(COLLECTION).as_posix(),
        'files': records, 'all_copied_files_sha256_verified': True,
        'generated_archive_files': ['README.txt', 'index.html', 'delivery_manifest.json'],
    })
    history.parent.mkdir(exist_ok=True)
    # Both resolved directories were checked above, on the same collection filesystem.
    dest.rename(history)
    staging.rename(dest)
    assert all(sha(history / rel) == digest for rel, digest in prior_files.items())
    for name in ('manifest.json', 'index.html', 'README.txt'):
        shutil.copy2(COLLECTION / name, history / ('collection_' + name + '.before'))
    files = [p for p in dest.rglob('*') if p.is_file()]
    manifest['figures']['F10'] = {
        'description': description, 'current_source': SOURCE.name,
        'status': 'updated_per_user_request', 'updated_on': DATE,
        'previous_version': history.relative_to(COLLECTION).as_posix(),
        'png': 'F10/F10.png', 'pdf': 'F10/F10.pdf', 'files': len(files),
        'bytes': sum(p.stat().st_size for p in files),
        'main_png_sha256': sha(dest / 'F10.png'), 'main_pdf_sha256': sha(dest / 'F10.pdf'),
    }
    manifest['last_updated_on'] = DATE
    write_json(COLLECTION / 'manifest.json', manifest)
    index = (COLLECTION / 'index.html').read_text(encoding='utf-8')
    section = ('<section><h2><a href="F10/index.html">F10</a></h2>'
               '<p>Case00 原始 / 重配置 3DGS：两行五列、五组严格同相机对比；保留确认视角与裁剪。' +
               ('下排使用已选定的双 Mask 优化 2000 步场景。' if DUALMASK else '') + '</p>'
               '<a href="F10/F10.png"><img src="F10/preview.jpg" loading="lazy"></a></section>')
    index, count = re.subn(r'<section><h2><a href="F10/index.html">.*?</section>', section, index, count=1, flags=re.S)
    assert count == 1
    write_text(COLLECTION / 'index.html', index)
    readme = (COLLECTION / 'README.txt').read_text(encoding='utf-8').rstrip()
    write_text(COLLECTION / 'README.txt', readme + '\n\nF10 update (' + DATE + '): ' + description +
               '\nPrevious files preserved in _history/' + history_name + '/.\n')
    assert all(sha(COLLECTION / rel) == digest for rel, digest in other_hashes.items())
    for page in (COLLECTION / 'index.html', dest / 'index.html'):
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
            assert (page.parent / link).is_file(), (page, link)
    print(json.dumps({'current': str(dest), 'backup': str(history), 'verified_source_copies': len(records),
                      'prior_files_preserved': len(prior_files), 'other_figures_unchanged': True,
                      'png_sha256': sha(dest / 'F10.png'), 'pdf_sha256': sha(dest / 'F10.pdf')}))


if __name__ == '__main__':
    main()
