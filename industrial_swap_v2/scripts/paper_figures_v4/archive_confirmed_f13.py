"""Archive the user-confirmed F13, preserving media and measured provenance."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
WITH_HEATMAP='--with-heatmap' in sys.argv
SOURCE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'/('F13_case02_registration_with_heatmap' if WITH_HEATMAP else 'F13_case02_registration_final')
EXPERIMENT=SOURCE.parent/'F13_case02_registration'
COLLECTION=ROOT/'experiment figure'
DEST=COLLECTION/'F13'
DATE='2026-11-12'
DESCRIPTION='Case02 registration: Coarse / Legacy / Gaussian-Aware; shared-camera renders, IoU, RGB error and enlarged workspace footprints with true centers and directions.'
if WITH_HEATMAP:
    DESCRIPTION+=' Includes the measured 61 x 37 yaw-scale full-objective slice with selected solution and sampled minimum.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    assert SOURCE.is_dir() and COLLECTION.is_dir()
    assert not DEST.is_symlink()
    assert DEST.resolve().parent==COLLECTION.resolve()
    if DEST.exists():
        assert WITH_HEATMAP, 'Existing confirmed F13 must not be overwritten without the requested replacement.'
        previous=json.loads((DEST/'delivery_manifest.json').read_text(encoding='utf-8'))
        assert previous['approved_source']=='industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration_final'
        assert sha(DEST/'F13.png')==sha(SOURCE.parent/'F13_case02_registration_final/figure_F13_final.png')
        assert sha(DEST/'F13.pdf')==sha(SOURCE.parent/'F13_case02_registration_final/figure_F13_final.pdf')
    manifest_path=COLLECTION/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    assert WITH_HEATMAP or 'F13' not in manifest['figures']
    old={r[k]:sha(COLLECTION/r[k]) for key,r in manifest['figures'].items() if key!='F13' for k in ('png','pdf')}
    index_path=COLLECTION/'index.html';index=index_path.read_text(encoding='utf-8')
    assert WITH_HEATMAP or 'F13/index.html' not in index
    assert '<a href="F17/index.html">F17</a>' in index
    assert '<section><h2><a href="F17/index.html">' in index
    records=[]
    renames={'figure_F13_final.png':'F13.png','figure_F13_final.pdf':'F13.pdf'}
    existing={p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()} if DEST.exists() else set()
    expected={renames.get(p.relative_to(SOURCE).as_posix(),p.relative_to(SOURCE).as_posix())
              for p in SOURCE.rglob('*') if p.is_file()}
    expected.update({'README.txt','index.html','delivery_manifest.json','workspace_footprint.png',
                     'data/registration_metrics.csv','data/registration_transforms.json','data/fixed_inputs.json'})
    assert existing<=expected, 'Unexpected old files: inspect before replacement.'
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        digest=sha(src);assert sha(dst)==digest
        records.append({'file':dst.relative_to(COLLECTION).as_posix(),
                        'source':src.relative_to(ROOT).as_posix(),'sha256':digest,'bytes':dst.stat().st_size})
    for src in sorted(SOURCE.rglob('*')):
        if src.is_file() and src.relative_to(SOURCE).as_posix() not in ('README.txt','index.html','delivery_manifest.json'):
            rel=src.relative_to(SOURCE).as_posix();copy(src,DEST/renames.get(rel,rel))
    for name in ('registration_metrics.csv','registration_transforms.json','fixed_inputs.json'):
        copy(EXPERIMENT/name,DEST/'data'/name)
    if WITH_HEATMAP:
        copy(SOURCE/'panels/workspace_footprint.png',DEST/'workspace_footprint.png')
    (DEST/'README.txt').write_text(
        'F13 - user-confirmed version, '+DATE+'\n'+DESCRIPTION+'\n\n'
        'F13.png / F13.pdf: confirmed publication figure, no version suffix.\n'
        'panels/: three unchanged scene images, two metric charts, footprint and (when present) yaw-scale objective.\n'
        'workspace_footprint.png: standalone workspace comparison.\n'
        'caption.txt / sources.json: caption and actual data/camera provenance.\n'
        'data/: original measured metrics, transforms and fixed inputs.\n'
        'code/: figure script snapshot; references retain their original experiment paths.\n'
        'delivery_manifest.json: source mapping and SHA256 verification.\n\n'
        'Only the Overflow chart was removed from the approved layout; the complete original '
        'evaluation is preserved in data/registration_metrics.csv. No experiments, models or media were modified by archiving.\n'+
        ('The yaw-scale grid contains 2257 actual full-objective evaluations. Raw CSV/NPZ, grid_summary.json, records/ and run.log are included. '
         'The earlier six-panel F13 has been replaced in this collection; its source experiment is preserved outside the collection.\n' if WITH_HEATMAP else ''),encoding='utf-8')
    (DEST/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>F13 - Confirmed</title>'
        '<style>body{font-family:Arial,sans-serif;max-width:1600px;margin:24px auto;padding:0 20px;color:#222}'
        'img{width:100%;height:auto}a{color:#185b94}nav a{margin-right:20px}</style>'
        '<nav><a href="../index.html">All confirmed figures</a><a href="F13.png">PNG</a>'
        '<a href="F13.pdf">PDF</a><a href="caption.txt">Caption</a>'
        '<a href="workspace_footprint.png">Workspace footprint</a>'+
        ('<a href="yaw_scale_objective.png">Yaw-scale objective</a><a href="yaw_scale_objective.csv">Grid CSV</a>' if WITH_HEATMAP else '')+
        '<a href="sources.json">Sources</a></nav>'
        '<h1>F13 - Case02 Spatial Registration</h1><a href="F13.png"><img src="F13.png"></a>',encoding='utf-8')
    write(DEST/'delivery_manifest.json',{'figure':'F13','user_confirmed':True,'confirmed_on':DATE,
        'description':DESCRIPTION,'approved_source':SOURCE.relative_to(ROOT).as_posix(),
        'files':records,'all_copied_files_sha256_verified':True,'no_media_content_edits':True,
        'generated_archive_files':['README.txt','index.html','delivery_manifest.json']})
    files=[p for p in DEST.rglob('*') if p.is_file()]
    manifest['figures']['F13']={'description':DESCRIPTION,'approved_source':SOURCE.name,
        'confirmed_on':DATE,'png':'F13/F13.png','pdf':'F13/F13.pdf','files':len(files),
        'bytes':sum(p.stat().st_size for p in files),'main_png_sha256':sha(DEST/'F13.png'),
        'main_pdf_sha256':sha(DEST/'F13.pdf')}
    manifest['figures']=dict(sorted(manifest['figures'].items()));manifest['last_updated_on']=DATE
    write(manifest_path,manifest)
    if not WITH_HEATMAP:
        index=index.replace('F11、F12、F17','F11、F12、F13、F17')
        index=index.replace('<a href="F17/index.html">F17</a>',
                            '<a href="F13/index.html">F13</a><a href="F17/index.html">F17</a>',1)
    section=('<section><h2><a href="F13/index.html">F13</a></h2>'
        '<p>Case02 空间配准：Coarse、Legacy、Gaussian-Aware；同相机结果、IoU、RGB Error、工位占据'+
        ('及真实 Yaw–Scale 完整目标函数热力图。' if WITH_HEATMAP else '。')+'</p>'
        '<a href="F13/F13.png"><img src="F13/F13.png" loading="lazy"></a></section>')
    if WITH_HEATMAP:
        index,n=re.subn(r'<section><h2><a href="F13/index.html">.*?</section>',section,index,count=1,flags=re.S)
        assert n==1
    else:
        index=index.replace('<section><h2><a href="F17/index.html">',section+'<section><h2><a href="F17/index.html">',1)
    index_path.write_text(index,encoding='utf-8')
    readme=COLLECTION/'README.txt'
    content=re.sub(r'^F13[^\n]*\n?','',readme.read_text(encoding='utf-8'),flags=re.M).rstrip()
    readme.write_text(content+'\n\nF13 ('+DATE+'): '+DESCRIPTION+'\n',encoding='utf-8')
    assert all(sha(COLLECTION/p)==digest for p,digest in old.items())
    print(json.dumps({'directory':str(DEST),'verified_copies':len(records),
        'other_confirmed_figures_unchanged':True,'png_sha256':sha(DEST/'F13.png'),'pdf_sha256':sha(DEST/'F13.pdf')}))


if __name__=='__main__':main()
