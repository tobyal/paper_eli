"""Add shared task captions to the accepted F07 image matrix. Layout only."""
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from relayout_f06_f07 import (ROOT, V3, V4, WIDTH, MARGIN, START, CELL, COL_GAP,
    CASES, METHODS, METHOD_TITLES, case_label, title, canvas_for, export, record, sha, dump)

OUT = V4/'F07_with_prompts'
PRIOR = V4/'F07_relayout'
TASKS = {
    'case00': 'Replace the industrial machine with a yellow six-axis robotic arm.',
    'case02': 'Replace the white wheeled machine with a stainless-steel roller conveyor.',
    'case04': 'Replace the industrial air cooler with a compact industrial process chiller.',
    'case07': 'Replace the stacked storage crates with an automated parts-storage cabinet.',
    'case10': 'Replace the industrial air cooler with a slim industrial tool cabinet.',
}
pdfmetrics.registerFont(TTFont('PaperItalic','/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'))
METHOD_FONT, PROMPT_FONT, PROMPT_BAND = 8., 7.8, 16.


def layout(cases,stem):
    inner=1.5
    locw=(CELL-inner)*.35
    resw=(CELL-inner)*.65
    header,rowgap=28.,5.
    rowh=PROMPT_BAND+resw
    height=2*MARGIN+header+len(cases)*rowh+(len(cases)-1)*rowgap
    dest=OUT/stem
    c=canvas_for(dest,height)
    c.setTitle('Qualitative comparison under shared industrial equipment replacement tasks')
    for j,name in enumerate(METHOD_TITLES):
        x=START+j*(CELL+COL_GAP)
        title(c,name,x,height-MARGIN-7,CELL,size=METHOD_FONT)
        title(c,'Loc.',x,height-MARGIN-20,locw,size=6.8)
        title(c,'Result',x+locw+inner,height-MARGIN-20,resw,size=6.8)
    task_boxes=[]
    for i,case in enumerate(cases):
        rowtop=height-MARGIN-header-i*(rowh+rowgap)
        text='"'+TASKS[case]+'"'
        span=4*CELL+3*COL_GAP
        textwidth=pdfmetrics.stringWidth(text,'PaperItalic',PROMPT_FONT)
        assert textwidth < span-4, 'Task text must fit on one line'
        c.setFillColorRGB(.10,.10,.10)
        c.setFont('PaperItalic',PROMPT_FONT)
        c.drawCentredString(START+span/2,rowtop-8,text)
        y=rowtop-rowh
        case_label(c,case,y,resw,letter=True)
        for j,method in enumerate(METHODS):
            x=START+j*(CELL+COL_GAP)
            folder=OUT/'panels'/case
            c.drawImage(str(folder/(method+'_localization.png')),x,y+(resw-locw)/2,locw,locw)
            c.drawImage(str(folder/(method+'_final.png')),x+locw+inner,y,resw,resw)
        task_boxes.append(dict(case=case,text=text,center_x_points=START+span/2,
                               baseline_y_points=rowtop-8,width_points=textwidth,
                               allowed_span_points=span,lines=1))
    result=export(c,dest)
    result.update(cases=cases,height_mm=height/72*25.4,width_mm=180,
        localization_width_mm=locw/72*25.4,result_width_mm=resw/72*25.4,
        inner_width_ratio=[.35,.65],method_font_pt=METHOD_FONT,prompt_font_pt=PROMPT_FONT,
        prompt_font='DejaVu Sans Oblique (embedded vector italic)',
        prompt_bands=task_boxes,source_pixels_and_aspect_ratios_preserved=True,
        new_crops=False,no_total_title=True,no_footer=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[p for root in [V3,V4] for p in root.glob('*/figure_*')
               if p.is_file() and p.parent != OUT]
    before={str(p):sha(p) for p in protected}
    previous=json.loads((PRIOR/'sources.json').read_text())
    prompt_path=ROOT/'baseline_comparison/prompts.json'
    prompts=json.loads(prompt_path.read_text())
    source_rows={r['case']:r for r in prompts['cases']}
    task_records={}
    copies={}
    for case in CASES:
        row=source_rows[case]
        task_records[case]=dict(display_instruction=TASKS[case],full_A=row['A'],full_B=row['B'],
            display_role='Condensed common replacement task, not a claim of identical internal model prompts.')
        folder=OUT/'panels'/case;folder.mkdir(parents=True,exist_ok=True)
        copies[case]={}
        for method in METHODS:
            cameras=previous['accepted_sources']['cases'][case]['camera_params']['methods'][method]
            main_cam=cameras.get('views',cameras)['main']
            assert main_cam['localization_camera']==main_cam['final_camera']
            for stage in ['localization','final']:
                filename=method+'_'+stage+'.png'
                src=PRIOR/'panels'/case/filename;dst=folder/filename
                assert sha(src)==previous['panels'][case][filename]['output']['sha256']
                shutil.copy2(src,dst)
                assert sha(dst)==sha(src)
                copies[case][filename]=dict(source=record(src),output=record(dst),byte_identical=True)
    layouts=dict(full=layout(CASES,'figure_F07_with_prompts'),
                 partA=layout(['case00','case02','case07'],'figure_F07_partA'),
                 partB=layout(['case04','case10'],'figure_F07_partB'))
    shared=('Columns compare GaussianEditor, DGE, GSLE Coarse, and our method. '
        'The centered italic instruction above each row is a concise description of the shared '
        'replacement task, derived from that case\'s recorded original-object (A) and replacement-object '
        '(B) descriptions. It is shown once for all four methods and does not imply identical internal '
        'model prompts: historical method-specific prompt adaptations and conditioning inputs are '
        'unchanged. Full A/B texts and source records are provided in prompts.txt and sources.json. '
        'Each method cell pairs a smaller localization view (left) with a larger replacement view '
        '(right), using a 35:65 width ratio. Red annotations show each method\'s actual selected '
        'Gaussians. Localization and replacement share the accepted camera within each method; '
        'cross-method viewpoints may differ. GSLE shows its Coarse 6500-step scene before DI-SDS. '
        'Our method additionally uses a registered reference photograph for instance specification. '
        'The comparison retains the accepted v8 views, original crops, real generation failures, '
        'and occlusion artifacts. No model rendering, generation, optimization, pixel retouching, '
        'camera adjustment, or Gaussian modification was performed.\n')
    row_descriptions={
        'full': 'Rows show (a) Case00, robot; (b) Case02, conveyor; (c) Case04, chiller; (d) Case07, storage cabinet; and (e) Case10, tool cabinet. ',
        'partA': 'Rows show (a) Case00, robot; (b) Case02, conveyor; and (d) Case07, storage cabinet. ',
        'partB': 'Rows show (c) Case04, chiller; and (e) Case10, tool cabinet. ',
    }
    for variant,rows in row_descriptions.items():
        suffix={'full':'','partA':'A','partB':'B'}[variant]
        filename='caption.txt' if variant=='full' else f'caption_{variant}.txt'
        (OUT/filename).write_text(f'Fig. 7{suffix}. Qualitative comparison with existing 3D Gaussian editing methods under shared industrial replacement tasks. '+rows+shared,encoding='utf-8')
    text=['Shared industrial equipment replacement tasks',
          'Display instructions are shortened task labels, not replacements for recorded method-specific prompts.',
          'A/B source: '+str(prompt_path),'Source SHA256: '+sha(prompt_path),'']
    for case in CASES:
        r=task_records[case]
        text += [case,'Display: "'+r['display_instruction']+'"','Full Text A: '+r['full_A'],
                 'Full Text B: '+r['full_B'],'']
    (OUT/'prompts.txt').write_text('\n'.join(text),encoding='utf-8')
    dump(OUT/'sources.json',dict(prior_manifest=record(PRIOR/'sources.json'),
        prompt_source=record(prompt_path),tasks=task_records,panels=copies,layouts=layouts,
        accepted_experiment_sources=previous['accepted_sources'],
        camera_changes=0,render_calls=0,new_model_calls=0,training_steps=0,new_crops=0,
        script=record(Path(__file__))))
    assert before=={str(p):sha(p) for p in protected},'Previous figure changed'
    dump(OUT/'validation.json',dict(old_figures_unchanged=True,protected_sha256=before,
        source_panels_byte_identical=True,paired_cameras_identical=True,gsle_stage='Coarse 6500',
        print_width_mm=180,method_font_pt=8,prompt_font_pt=7.8,prompt_italic=True,
        prompt_single_line=True,source_image_display_sizes_unchanged=True,
        note='Parts A/B provide pagination options; at 180 mm their individual images are the same size as the complete version.'))
    body=''
    for stem,label in [('figure_F07_with_prompts','五案例完整版'),('figure_F07_partA','Part A: Case00 / Case02 / Case07'),('figure_F07_partB','Part B: Case04 / Case10')]:
        body+=f'<section><h2>{label}</h2><p><a href="{stem}.png">300 DPI PNG</a> · <a href="{stem}.pdf">PDF</a></p><img src="{stem}_preview.jpg"></section>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>F07 Shared Text Instructions</title>'
        '<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}section{margin:28px 0}</style>'
        '<h1>F07 · 共享文本输入版</h1><p>每行斜体文本表达共同替换任务；各方法内部 Prompt 和实验结果保持原样。</p>'
        '<p><a href="prompts.txt">任务文本与完整 A/B</a> · <a href="caption.txt">Caption</a> · '
        '<a href="sources.json">来源记录</a> · <a href="../F07_with_prompts.zip">完整下载包</a></p>'+body,encoding='utf-8')
    with ZipFile(V4/'F07_with_prompts.zip','w',ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps({k:dict(pixels=v['pixels'],width_mm=v['width_mm'],height_mm=v['height_mm'])
                      for k,v in layouts.items()}),flush=True)


if __name__=='__main__':main()
