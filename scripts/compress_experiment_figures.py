"""Compress embedded scene images while keeping PDF text, vectors and pages."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/pdf/experiment_figures.pdf'
OUTPUT = ROOT / 'output/pdf/experiment_figures_compressed.pdf'
REPORT = ROOT / 'tmp/pdfs/experiment_figures_compressed/compression.json'


def main():
    assert not OUTPUT.exists(), 'Keep any existing compressed deliverable.'
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original = PdfReader(SOURCE)
    writer = PdfWriter(clone_from=SOURCE)
    seen = set()
    changed = []
    for number, page in enumerate(writer.pages, 1):
        for image in page.images:
            ref = image.indirect_reference
            if ref is None or ref.idnum in seen:
                continue
            seen.add(ref.idnum)
            obj = ref.get_object()
            # Keep masks, transparency, palettes and small diagrams lossless.
            if (obj.get('/ColorSpace') != '/DeviceRGB' or '/SMask' in obj
                    or '/Mask' in obj or max(obj['/Width'], obj['/Height']) < 600):
                continue
            pixels = image.image
            if pixels.mode != 'RGB':
                continue
            before = pixels.size
            pixels.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
            image.replace(pixels, quality=85, subsampling=0, optimize=True)
            changed.append({'page': number, 'name': image.name,
                            'before_pixels': before, 'after_pixels': pixels.size})
        page.compress_content_streams()
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    writer.add_metadata({'/Title': 'Experiment Figures - Compressed'})
    writer.write(OUTPUT)
    writer.close()
    result = PdfReader(OUTPUT)
    assert len(result.pages) == len(original.pages) == 12
    for old, new in zip(original.pages, result.pages):
        assert list(old.mediabox) == list(new.mediabox)
        assert old.extract_text() == new.extract_text()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    report = {'source': str(SOURCE), 'source_sha256': source_hash, 'output': str(OUTPUT),
              'source_bytes': SOURCE.stat().st_size, 'output_bytes': OUTPUT.stat().st_size,
              'reduction_percent': round(100 * (1 - OUTPUT.stat().st_size / SOURCE.stat().st_size), 2),
              'pages': len(result.pages), 'text_and_vector_layout_preserved': True,
              'jpeg_quality': 85, 'chroma_subsampling': 0, 'changed_images': changed}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'changed_images'}))
    print('Compressed embedded images:', len(changed))


if __name__ == '__main__':
    main()
