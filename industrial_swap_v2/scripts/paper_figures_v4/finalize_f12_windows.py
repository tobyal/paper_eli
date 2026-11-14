"""Finish a verified F12 publication when Windows holds the directory open."""
import json
import re
import shutil
from pathlib import Path

from archive_confirmed_f14_f15 import sha, write_json, write_text

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / 'experiment figure'
DEST = COLLECTION / 'F12'
STAGE = COLLECTION / '.F12_depth_subset_staging'
HISTORY = COLLECTION / '_history/F12_20261113_before_depth_subset'
METADATA = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F12_depth_four_case_summary/collection_metadata'


def main():
    for directory in (DEST, STAGE, HISTORY):
        assert not directory.is_symlink()
        directory.resolve().relative_to(COLLECTION.resolve())
    assert DEST.is_dir() and STAGE.is_dir() and not HISTORY.exists()
    published = json.loads((METADATA / 'manifest.json').read_text(encoding='utf-8-sig'))
    old = json.loads((COLLECTION / 'manifest.json').read_text(encoding='utf-8-sig'))
    other = {record[key]: sha(COLLECTION / record[key])
             for name, record in old['figures'].items() if name != 'F12'
             for key in ('png', 'pdf')}
    assert sha(STAGE / 'F12.png') == published['figures']['F12']['main_png_sha256']
    assert sha(STAGE / 'F12.pdf') == published['figures']['F12']['main_pdf_sha256']
    prior = {p.relative_to(DEST): sha(p) for p in DEST.rglob('*') if p.is_file()}
    updates = {p.relative_to(STAGE): sha(p) for p in STAGE.rglob('*') if p.is_file()}
    for panel in ('a', 'c', 'd'):
        assert prior[Path('panels') / (panel + '.png')] == updates[Path('panels') / (panel + '.png')]
    shutil.copytree(DEST, HISTORY)
    assert all(sha(HISTORY / rel) == digest for rel, digest in prior.items())
    for name in ('manifest.json', 'index.html', 'README.txt'):
        shutil.copy2(COLLECTION / name, HISTORY / ('collection_' + name + '.before'))
    for rel, digest in updates.items():
        target = DEST / rel
        target.resolve().relative_to(DEST.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGE / rel, target)
        assert sha(target) == digest
    for rel in set(prior) - set(updates):
        target = DEST / rel
        target.resolve().relative_to(DEST.resolve())
        assert sha(target) == sha(HISTORY / rel)
        target.unlink()
    old['figures']['F12'] = published['figures']['F12']
    old['last_updated_on'] = published['last_updated_on']
    write_json(COLLECTION / 'manifest.json', old)
    pattern = r'<section><h2><a href="F12/index.html">.*?</section>'
    section = re.search(pattern, (METADATA / 'index.html').read_text(encoding='utf-8'), re.S).group(0)
    index, count = re.subn(pattern, lambda _: section,
                          (COLLECTION / 'index.html').read_text(encoding='utf-8'), count=1, flags=re.S)
    assert count == 1
    write_text(COLLECTION / 'index.html', index)
    readme = (COLLECTION / 'README.txt').read_text(encoding='utf-8').rstrip()
    write_text(COLLECTION / 'README.txt', readme + '\n\nF12 update (2026-11-13): '
               + published['figures']['F12']['description'] + '\n')
    for page in (COLLECTION / 'index.html', DEST / 'index.html'):
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding='utf-8')):
            assert (page.parent / link).is_file()
    assert all(sha(HISTORY / rel) == digest for rel, digest in prior.items())
    assert all(sha(COLLECTION / rel) == digest for rel, digest in other.items())
    print(json.dumps({'current': str(DEST), 'backup': str(HISTORY),
                      'png_sha256': sha(DEST / 'F12.png'), 'pdf_sha256': sha(DEST / 'F12.pdf'),
                      'unchanged_panels': ['a', 'c', 'd'], 'files_verified': len(updates)}))


if __name__ == '__main__':
    main()
