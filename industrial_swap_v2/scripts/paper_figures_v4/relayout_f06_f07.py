"""Publication-only layout; saved images/cameras/models remain untouched.

Run with remote system python3. F07 is built first, followed by F06.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

from figure_f05_revised import ROOT, V3, V4, sha, record, dump

WIDTH = 180/25.4*72
MARGIN, LABEL_W, LABEL_GAP = 3., 58., 3.
COL_GAP = 3.
START = MARGIN+LABEL_W+LABEL_GAP
CELL = (WIDTH-START-MARGIN-3*COL_GAP)/4
CASES = ['case00','case02','case04','case07','case10']
NAMES = ['Robot','Conveyor','Chiller','Storage Cabinet','Tool Cabinet']
METHODS = ['gaussianeditor','dge','gsle','ours']
METHOD_TITLES = ['GaussianEditor','DGE','GSLE Coarse','Ours']
FONT_SIZE, CASE_SIZE = 7.4, 7.0


def title(c,text,x,y,width,size=FONT_SIZE):
    assert pdfmetrics.stringWidth(text,'PaperSans',size)<=width,(text,width)
    c.setFillColorRGB(.06,.06,.06)
    c.setFont('PaperSans',size)
    c.drawCentredString(x+width/2,y,text)


def case_label(c,case,y,height,letter=False):
    i=CASES.index(case)
    c.setFillColorRGB(.06,.06,.06)
    c.setFont('PaperSans',CASE_SIZE)
    first=(f'({chr(97+i)}) ' if letter else '')+'Case'+case[-2:]
    for line,baseline in [(first,y+height/2+3),(NAMES[i],y+height/2-7)]:
        assert pdfmetrics.stringWidth(line,'PaperSans',CASE_SIZE)<=LABEL_W
        c.drawString(MARGIN,baseline,line)


def export(c,dest):
    c.showPage(); c.save()
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(dest.with_suffix('.pdf')),str(dest)],check=True)
    im=Image.open(dest.with_suffix('.png'))
    im.save(dest.with_suffix('.png'),dpi=(300,300))
    ImageOps.contain(im,(1800,2000)).save(dest.parent/(dest.name+'_preview.jpg'),quality=95)
    subprocess.run(['pdftoppm','-singlefile','-r','150','-png',str(dest.with_suffix('.pdf')),
                    str(dest.parent/(dest.name+'_180mm_proof'))],check=True)
    return dict(pdf=record(dest.with_suffix('.pdf')),png=record(dest.with_suffix('.png')),
                pixels=list(im.size),width_mm=180,dpi=300)


def canvas_for(dest,height):
    c=canvas.Canvas(str(dest.with_suffix('.pdf')),pagesize=(WIDTH,height),pageCompression=1)
    c.setAuthor('paper_eli - existing experimental image layout')
    return c


def f07_layout(out,cases,stem):
    inner=1.5
    locw=(CELL-inner)*.35
    resw=(CELL-inner)*.65
    header,rowgap=28.,5.
    height=2*MARGIN+header+len(cases)*resw+(len(cases)-1)*rowgap
    dest=out/stem
    c=canvas_for(dest,height)
    c.setTitle('Qualitative comparison with existing 3D Gaussian editing methods')
    for j,name in enumerate(METHOD_TITLES):
        x=START+j*(CELL+COL_GAP)
        title(c,name,x,height-MARGIN-7,CELL)
        title(c,'Loc.',x,height-MARGIN-20,locw,size=6.8)
        title(c,'Result',x+locw+inner,height-MARGIN-20,resw,size=6.8)
    for i,case in enumerate(cases):
        y=height-MARGIN-header-resw-i*(resw+rowgap)
        case_label(c,case,y,resw,letter=True)
        for j,method in enumerate(METHODS):
            x=START+j*(CELL+COL_GAP)
            p=out/'panels'/case
            # Source views are square. Both retain every source pixel, including failures.
            c.drawImage(str(p/(method+'_localization.png')),x,y+(resw-locw)/2,locw,locw)
            c.drawImage(str(p/(method+'_final.png')),x+locw+inner,y,resw,resw)
    meta=export(c,dest)
    meta.update(cases=cases,height_mm=height/72*25.4,localization_width_mm=locw/72*25.4,
                result_width_mm=resw/72*25.4,internal_width_ratio=[.35,.65],
                height_points=height,no_new_crop=True,case_label_lines=2,
                minimum_font_pt=6.8,source_images_not_stretched=True)
    return meta


def f07():
    old=V3/'F07_comparison'
    out=V4/'F07_relayout';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((old/'comparison_sources.json').read_text())
    files={}
    for case in CASES:
        p=out/'panels'/case;p.mkdir(parents=True,exist_ok=True)
        files[case]={}
        for method in METHODS:
            cams=meta['cases'][case]['camera_params']['methods'][method]
            main=cams.get('views',cams)['main']
            assert main['localization_camera']==main['final_camera']
            for stage in ['localization','final']:
                name=method+'_'+stage+'.png'
                src=old/'panels'/case/name
                dst=p/name
                assert sha(src)==meta['cases'][case]['input_files'][name]['sha256']
                shutil.copy2(src,dst)
                assert sha(src)==sha(dst)
                assert Image.open(dst).size[0]==Image.open(dst).size[1]
                files[case][name]=dict(source=record(src),output=record(dst),byte_identical=True)
    full=f07_layout(out,CASES,'figure_F07_revised')
    # Four methods at 180 mm leave ~25 mm per result. Provide requested pagination
    # alternatives without claiming that splitting changes the per-cell print width.
    parta=f07_layout(out,['case00','case02','case07'],'figure_F07_partA')
    partb=f07_layout(out,['case04','case10'],'figure_F07_partB')
    caption=('Fig. 7. Qualitative comparison with existing 3D Gaussian editing methods. '
        'Rows show (a) Case00, robot; (b) Case02, conveyor; (c) Case04, chiller; '
        '(d) Case07, storage cabinet; and (e) Case10, tool cabinet. Columns compare '
        'GaussianEditor, DGE, GSLE Coarse, and our method. Each method cell pairs a smaller '
        'localization view on the left with a larger replacement view on the right '
        '(35:65 width ratio). Translucent red marks the original Gaussians selected by that '
        'method. Localization and replacement retain exactly the same accepted camera within '
        'each method; cross-method cameras may differ, as in the accepted v8 comparison. '
        'GSLE uses the saved Coarse 6500-step scene before DI-SDS. Localization inputs differ: '
        'the baselines use their native localization, while our method additionally uses a '
        'registered reference photograph. This is a viewpoint-matched qualitative comparison, '
        'not an identical-camera, identical-input pixel comparison. All saved views, crops, '
        'real generation failures, and occlusion artifacts are retained without retouching. '
        'No rendering, training, camera adjustment, or Gaussian modification was performed.\n')
    (out/'caption.txt').write_text(caption,encoding='utf-8')
    dump(out/'sources.json',dict(source_manifest=record(old/'comparison_sources.json'),
        accepted_sources=meta,panels=files,layouts=dict(full=full,partA=parta,partB=partb),
        renderer_runs=0,training_steps=0,camera_changes=0,new_crops=0,retouching=False,
        gsle_stage='Coarse 6500',script=record(Path(__file__))))
    dump(out/'readability.json',dict(width_mm=180,label_font_pt=7.4,case_font_pt=7,
        locator_label_font_pt=6.8,result_width_mm=full['result_width_mm'],
        loc_width_mm=full['localization_width_mm'],
        assessment='Complete figure is an overview; fine mechanical defects require the source panels or PDF zoom. '
                   'Splits provide pagination flexibility, not a larger per-cell size at identical 180 mm print width.',
        split_reason='Four method cells constrain result width to approximately 25 mm; retain full overview and provide requested alternatives.'))
    print('F07 COMPLETE',full['pixels'],flush=True)


def fit_asset(src,dst):
    im=Image.open(src).convert('RGB')
    a=np.asarray(im)
    yy,xx=np.where(np.any(a!=255,axis=2))
    assert len(xx)
    # Include every non-white pixel; no connected-component filtering or cleanup.
    bbox=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
    content=im.crop(bbox)
    size=(1600,1120)
    scale=min(size[0]*.82/content.width,size[1]*.82/content.height)
    newsize=(round(content.width*scale),round(content.height*scale))
    content=content.resize(newsize,getattr(Image,'Resampling',Image).LANCZOS)
    result=Image.new('RGB',size,'white')
    offset=((size[0]-newsize[0])//2,(size[1]-newsize[1])//2)
    result.paste(content,offset)
    result.save(dst,dpi=(300,300))
    occ=max(newsize[0]/size[0],newsize[1]/size[1])
    assert .75<=occ<=.85
    return dict(original_nonwhite_bbox=bbox,all_nonwhite_pixels_retained=True,
                canvas_size=size,scaled_content_size=newsize,offset=offset,
                dominant_axis_canvas_occupancy=occ,
                operation='Only uniform resizing and white-margin reframing; no asset or camera changes.')


def f06():
    old=V3/'F06_five_cases'
    out=V4/'F06_relayout';out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((old/'case_manifest.json').read_text())
    keys=['original','m2_reference','m3_gaussian_asset','m4_integrated_scene']
    columns=['Original Scene','2D Reference','3D Asset','Reconfigured Scene']
    files={}
    for case in CASES:
        assert next(r for r in prior['cases'] if r['case']==case)['same_original_final_camera']
        folder=out/'panels'/case;folder.mkdir(parents=True,exist_ok=True)
        files[case]={}
        for key in keys:
            src=old/'panels'/case/(key+'.png');dst=folder/(key+'.png')
            if key=='m3_gaussian_asset':
                presentation=fit_asset(src,dst)
            else:
                im=Image.open(src).convert('RGB');w,h=im.size
                assert w==h
                # Identical normalized vertical display crop across the three scene stages.
                crop=(0,round(h*.15),w,round(h*.85))
                result=im.crop(crop);result.save(dst,dpi=(300,300))
                assert Image.open(dst).tobytes()==result.tobytes()
                presentation=dict(crop_pixels=crop,normalized_crop=[0,.15,1,.85],
                                  retained_pixels_exact=True)
            files[case][key]=dict(source=record(src),output=record(dst),presentation=presentation)
    header,rowgap=18.,3.5
    cellh=CELL*.7
    height=2*MARGIN+header+5*cellh+4*rowgap
    dest=out/'figure_F06_revised';c=canvas_for(dest,height)
    c.setTitle('Stage-wise results of text-guided industrial equipment replacement')
    for j,label in enumerate(columns):
        title(c,label,START+j*(CELL+COL_GAP),height-MARGIN-8,CELL)
    for i,case in enumerate(CASES):
        y=height-MARGIN-header-cellh-i*(cellh+rowgap)
        case_label(c,case,y,cellh)
        for j,key in enumerate(keys):
            x=START+j*(CELL+COL_GAP)
            p=out/'panels'/case/(key+'.png')
            iw,ih=Image.open(p).size
            scale=min(CELL/iw,cellh/ih)
            dw,dh=iw*scale,ih*scale
            c.drawImage(str(p),x+(CELL-dw)/2,y+(cellh-dh)/2,dw,dh)
            if key=='m3_gaussian_asset':
                c.setStrokeColorRGB(.91,.91,.91);c.setLineWidth(.22)
                c.rect(x,y,CELL,cellh,fill=0,stroke=1)
    layout=export(c,dest)
    layout.update(height_mm=height/72*25.4,rows=5,columns=4,column_titles=columns,
        case_label_lines=2,cell_mm=[CELL/72*25.4,cellh/72*25.4],
        no_title=True,no_footer=True,fonts_pt=[FONT_SIZE,CASE_SIZE])
    caption=('Fig. 6. Stage-wise results of text-guided industrial equipment replacement. '
        'Rows show Case00 (robot), Case02 (conveyor), Case04 (chiller), Case07 (storage cabinet), '
        'and Case10 (tool cabinet). Columns show the original 3DGS scene, the saved text-guided '
        'Qwen 2D reference, the independent Gaussian asset trained with PBR multi-view supervision, '
        'and the reconfigured scene after spatial alignment and local-background optimization. '
        'Within each case, original and reconfigured scenes use their previously saved identical '
        'camera and identical display crop. The three scene columns use the same normalized '
        'vertical display bounds, preserving the complete target device and avoiding stretching. '
        'The asset column uses the original white-background render with only uniform presentation '
        'scaling and white-margin adjustment; the largest normalized object extent occupies about '
        '82% of its canvas. All non-white asset pixels are included. Case07 preserves the accepted '
        'yaw adjustment. All source scene versions match the original F06; the previously selected '
        'optimization candidates are not relabeled as automatic guard-passed exports. '
        'No generation, rendering, training, image retouching, or camera/geometry changes were performed.\n')
    (out/'caption.txt').write_text(caption,encoding='utf-8')
    dump(out/'sources.json',dict(source_manifest=record(old/'case_manifest.json'),
        original_manifest=prior,panels=files,layout=layout,script=record(Path(__file__)),
        same_original_final_camera_and_crop=True,model_runs=0,render_runs=0,image_retouching=False))
    dump(out/'readability.json',dict(width_mm=180,label_font_pt=FONT_SIZE,case_font_pt=CASE_SIZE,
        panel_width_mm=CELL/72*25.4,panel_height_mm=cellh/72*25.4,
        assessment='Compact landscape 5-by-4 overview. Two-line case labels keep the left margin narrow and readable; image cells are identical.'))
    print('F06 COMPLETE',layout['pixels'],flush=True)


def package():
    for fig in ['F06','F07']:
        out=V4/(fig+'_relayout')
        source=json.loads((out/'sources.json').read_text())
        source['script']=record(Path(__file__))
        dump(out/'sources.json',source)
        if fig=='F07':
            caption=(out/'caption.txt').read_text()
            start=caption.index('Rows show ')
            end=caption.index('Columns compare ')
            for letter,rows in [('A','Rows show (a) Case00, robot; (b) Case02, conveyor; and (d) Case07, storage cabinet. '),
                                ('B','Rows show (c) Case04, chiller; and (e) Case10, tool cabinet. ')]:
                text=(caption[:start]+rows+caption[end:]).replace('Fig. 7.',f'Fig. 7{letter}.',1)
                (out/f'caption_part{letter}.txt').write_text(text,encoding='utf-8')
        stems=[f'figure_{fig}_revised']+(['figure_F07_partA','figure_F07_partB'] if fig=='F07' else [])
        body=''
        for stem in stems:
            body+=f'<section><p>{stem}: <a href="{stem}.png">300 DPI PNG</a> · <a href="{stem}.pdf">PDF</a> · <a href="{stem}_180mm_proof.png">180 mm / 150 DPI proof</a></p><img src="{stem}_preview.jpg"></section>'
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+fig+' relayout</title>'
            '<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}section{margin:24px 0}</style>'
            '<h1>'+fig+' · 论文排版检查</h1><p>仅重新排版已有高清素材，保留原版与相机。'
            '<a href="caption.txt">Caption</a> · <a href="sources.json">来源</a> · '
            '<a href="readability.json">印刷尺寸记录</a> · <a href="../F06_F07_relayout.zip">完整包</a></p>'+body,encoding='utf-8')
    with ZipFile(V4/'F06_F07_relayout.zip','w',ZIP_DEFLATED,compresslevel=4) as z:
        for folder in [V4/'F06_relayout',V4/'F07_relayout']:
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(V4))


def main(resume_f07=False):
    protected=[p for root in [V3,V4] for p in root.glob('*/figure_*')
               if p.is_file() and p.parent.name not in ['F06_relayout','F07_relayout']]
    before={str(p):sha(p) for p in protected}
    if resume_f07:
        saved=V4/'F07_relayout/sources.json'
        prior=json.loads(saved.read_text())
        for layout in prior['layouts'].values():
            assert sha(layout['pdf']['path'])==layout['pdf']['sha256']
        prior['script']=record(Path(__file__))
        dump(saved,prior)
    else:
        f07()
    f06()
    assert before=={str(p):sha(p) for p in protected},'An existing figure changed'
    for folder in ['F06_relayout','F07_relayout']:
        dump(V4/folder/'validation.json',dict(old_figures_unchanged=True,
             old_figure_sha256=before,new_renderer_calls=0,new_model_calls=0))
    package()
    print('BOTH FIGURES PACKAGED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-f07',action='store_true')
    parser.add_argument('--package-only',action='store_true')
    args=parser.parse_args()
    if args.package_only:package()
    else:main(args.resume_f07)
