"""Merge the current figure collection, one original PDF figure per page."""
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / 'experiment figure'
OUTPUT = ROOT / 'output/pdf/experiment_figures.pdf'
RECORD = ROOT / 'tmp/pdfs/experiment_figures_combined/sources.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((COLLECTION / 'manifest.json').read_text(encoding='utf-8-sig'))
    figures = manifest['figures']
    actual = {p.name for p in COLLECTION.iterdir() if p.is_dir() and re.fullmatch(r'F\d+', p.name)}
    assert actual == set(figures), (actual, set(figures))
    names = sorted(figures, key=lambda name: int(name[1:]))
    writer = PdfWriter()
    records = []
    for number, name in enumerate(names, 1):
        source = COLLECTION / figures[name]['pdf']
        assert sha(source) == figures[name]['main_pdf_sha256'], source
        reader = PdfReader(source)
        assert len(reader.pages) == 1, (name, len(reader.pages))
        writer.append(reader, outline_item=name, import_outline=False)
        page = reader.pages[0]
        records.append({'page': number, 'figure': name, 'source': str(source),
                        'sha256': sha(source), 'media_box': list(map(float, page.mediabox)),
                        'content_sha256': hashlib.sha256(page.get_contents().get_data()).hexdigest()})
    writer.add_metadata({'/Title': 'Experiment Figures', '/Subject': 'Current experiment figure collection, one figure per page'})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT.exists():
        writer.write(OUTPUT)
    writer.close()
    merged = PdfReader(OUTPUT)
    assert len(merged.pages) == len(records)
    for page, record in zip(merged.pages, records):
        assert all(abs(float(x) - y) < 1e-5 for x, y in zip(page.mediabox, record['media_box']))
        assert hashlib.sha256(page.get_contents().get_data()).hexdigest() == record['content_sha256']
        assert sha(Path(record['source'])) == record['sha256']
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps({'output': str(OUTPUT), 'pages': records,
                                 'output_sha256': sha(OUTPUT)}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'pages': len(records), 'order': names,
                      'size_mb': round(OUTPUT.stat().st_size / 1024**2, 1),
                      'page_content_unchanged': True}))


if __name__ == '__main__':
    main()
