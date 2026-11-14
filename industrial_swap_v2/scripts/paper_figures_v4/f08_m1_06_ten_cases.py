"""Restore centered Input typography; summarize the 10 saved M1 case judgments."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
OUT=BASE/'F08_final_M1_06_ten_cases'
SOURCE=BASE/'F08_final_M1_06_compact'
STYLE=BASE/'F08_final_M1_06_adjusted'
AUDIT=BASE/'F08_localization_revised/instance_selection_results.csv'
CANDIDATE=BASE/'F08_candidate_selection/candidate_02'
RENDERS=OUT/'renders'
M1_RECORD=ROOT/'industrial_swap_v2/webapp/data/m1_assist/cases/a0ee4103b79a4fcb8911ffaecb1720aa/job.json'
MAIN_CROP=[196,0,938,510]  # The two comparison coolers remain clear of right-side insets.
TARGET_CROP=[226,180,360,314]
WRONG_CROP=[499,213,690,404]
SCALE=3

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    assert SOURCE.is_dir() and M1_RECORD.is_file()
    for folder in ['panels','inputs','code','renders']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    snapshot={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    job=read(M1_RECORD)
    text_a=job['text'].strip()
    assert text_a
    shutil.copy2(M1_RECORD,OUT/'inputs/m1_case06_job.json')
    shutil.copy2(AUDIT,OUT/'instance_selection.csv')
    shutil.copy2(CANDIDATE/'evidence/photo_assisted_visible_mask.png',OUT/'inputs/intended_target_projection.png')
    shutil.copy2(CANDIDATE/'original_reference/sam_mask.png',OUT/'inputs/original_reference_sam.png')
    shutil.copy2(CANDIDATE/'original_reference/reference_original.jpg',OUT/'inputs/original_reference.jpg')
    shutil.copy2(CANDIDATE/'selected_camera_original.jpg',OUT/'inputs/display_camera_original.jpg')
    mask=np.asarray(Image.open(OUT/'inputs/intended_target_projection.png'))>127
    # Restore every original irregular boundary. Only the exact detached line
    # circled by the user is omitted from the DISPLAY contour (12 pixels).
    removed_roi=[311,283,317,292]
    x0,y0,x1,y1=removed_roi
    display_contour=mask.astype(np.float32)
    assert int(mask[y0:y1,x0:x1].sum())==12
    display_contour[y0:y1,x0:x1]=0
    changed=mask!=(display_contour>.5)
    assert int(changed.sum())==12
    unchanged=changed.copy();unchanged[y0:y1,x0:x1]=False
    assert not unchanged.any()
    Image.fromarray((display_contour*255).astype(np.uint8)).save(OUT/'inputs/intended_contour_display_only.png')
    np.save(OUT/'inputs/intended_contour_display_field.npy',display_contour)
    smoothing=dict(display_only=True,original_mask_unchanged=True,no_smoothing=True,
        original_irregular_boundary_restored=True,removed_user_circled_roi=removed_roi,
        removed_display_pixels=12,all_other_contour_input_pixels_identical=True,
        original_pixels=int(mask.sum()),display_pixels=int((display_contour>.5).sum()),
        not_used_for_localization_or_metrics=True)
    shutil.copytree(SOURCE/'renders',RENDERS,dirs_exist_ok=True)
    with (OUT/'instance_selection.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    counts=[sum(r[k].lower()=='true' for r in rows) for k in ['text_only_correct_instance','photo_assisted_correct_instance']]
    n=len(rows)
    assert n==10 and counts==[4,10]
    assert len({r['physical_target_group'] for r in rows})==9
    # Reuse the earlier real splat renders with 60% red color mixing, exactly.
    image_sources={key:RENDERS/(key+'.png') for key in ['a_main','b_main','wrong_instance']}
    images={key:Image.open(p).convert('RGB') for key,p in image_sources.items()}

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    # Add exactly 72 pixels at 300 DPI above the unchanged previous layout.
    # This preserves the pixel grid for the accepted lower chart.
    w,h=180.,108.+72*25.4/300.  # Restore the earlier centered Input layout.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    def show(ax,key,crop,cyan):
        ax.imshow(images[key],extent=[-.5,937.5,527.5,-.5],interpolation='lanczos')
        if cyan:ax.contour(display_contour,levels=[.5],colors=['#00cfe7'],linewidths=.95,antialiased=True)
        ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5)
        ax.set_xticks([]);ax.set_yticks([])
    pw=87.;ph=pw*(MAIN_CROP[3]-MAIN_CROP[1])/(MAIN_CROP[2]-MAIN_CROP[0])
    target_inset_size=22.5
    wrong_inset_size=21.
    inset_header_height=2.8
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    input_line=HPacker(children=[
        TextArea('Input:',textprops=dict(fontsize=8.2,fontweight='bold',color='#222222')),
        TextArea('"'+text_a+'"',textprops=dict(fontsize=8.2,fontstyle='italic',color='#222222'))],
        align='center',pad=0,sep=4)
    input_anchor=AnchoredOffsetbox(loc='center',child=input_line,pad=0,frameon=False,
        bbox_to_anchor=(.5,(h-3.)/h),bbox_transform=fig.transFigure,borderpad=0)
    fig.add_artist(input_anchor)
    for col,(key,title) in enumerate([('a_main','(a) Text-only Localization'),('b_main','(b) Photo-guided Localization')]):
        x=1.5+90*col;y=41.
        ax=axis(x,y,pw,ph);show(ax,key,MAIN_CROP,True)
        for spine in ax.spines.values():spine.set_visible(False)
        text(x,y+ph+3.1,title,ha='left',va='center',fontsize=8.4)
        insets=[(x+pw-target_inset_size-1.2,y+ph-target_inset_size-inset_header_height-1.2,
            target_inset_size,key,TARGET_CROP,'Intended Target' if col==0 else 'Correct Target',True)]
        if col==0:
            insets.append((x+pw-wrong_inset_size-1.2,y+1.2,
                wrong_inset_size,'wrong_instance',WRONG_CROP,'Wrong Instance',False))
        for zx,zy,size,zkey,box,label,cyan in insets:
            ia=axis(zx,zy,size,size);show(ia,zkey,box,cyan)
            for spine in ia.spines.values():
                spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.8)
            # Fixed physical header height and font size across all inset sizes.
            header_fraction=inset_header_height/size
            ia.add_patch(Rectangle((0,1),1,header_fraction,transform=ia.transAxes,clip_on=False,
                facecolor='white',alpha=1.,edgecolor='white',linewidth=.8))
            ia.text(.5,1+header_fraction/2,label,transform=ia.transAxes,va='center',ha='center',
                fontsize=5.8,color='black',clip_on=False)
        images[key].crop(tuple(v*SCALE for v in MAIN_CROP)).save(OUT/'panels'/f'{key}_scene.png',dpi=(300,300))
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.8,label='Selected Gaussians'),
        Line2D([0],[0],color='#00cfe7',lw=1.5,label='Intended target')],loc='center',bbox_to_anchor=(.5,35.8/h),
        ncol=2,frameon=False,fontsize=6.7,columnspacing=1.5,handlelength=1.6)

    # Panel (c): unchanged chart style, now using all 10 saved M1 case judgments.
    text(1.5,31,'(c) Correct vs. Incorrect Instance Selection',ha='left',va='center',fontsize=8.1)
    correct='#196f86';incorrect='#d9a277'
    fig.legend(handles=[Patch(facecolor=correct,label='Correct'),Patch(facecolor=incorrect,label='Incorrect')],
        loc='center',bbox_to_anchor=(155/w,31/h),ncol=2,frameon=False,fontsize=6.1,
        handlelength=1.2,handletextpad=.4,columnspacing=1.2)
    ax=axis(36,11,139,16.5)
    values=np.array(counts)/n
    ax.barh([1,0],values,height=.62,color=correct,zorder=3)
    ax.barh([1,0],1-values,left=values,height=.62,color=incorrect,zorder=3)
    ax.text(values[0]/2,1,'4/10 (40%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(values[0]+(1-values[0])/2,1,'6/10 (60%)',ha='center',va='center',fontsize=7.1,color='#3f2b20',fontweight='bold')
    ax.text(.5,0,'10/10 (100%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(.985,0,'Incorrect: 0/10 (0%)',ha='right',va='center',fontsize=5.8,color='white')
    ax.set_yticks([1,0],['Text-only','Photo-guided (Ours)']);ax.tick_params(axis='y',length=0,labelsize=7.1,pad=5)
    ax.set_ylim(-.55,1.55);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(axis='x',labelsize=6.1,length=2,pad=2)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#777777')
    ax.grid(axis='x',color='#ededed',zorder=0)
    text(1.5,2.3,'10 M1 cases',ha='left',va='center',fontsize=5.9,color='#555555')
    text(178.5,2.3,'Preliminary AI visual audit',ha='right',va='center',fontsize=5.9,color='#555555')
    fig.savefig(OUT/'figure_F08_final.png',dpi=300);fig.savefig(OUT/'figure_F08_final.pdf')
    for col,name in enumerate(['a_text_only','b_photo_guided']):
        bounds=[.5+90*col,40,89.5+90*col,105.8]
        fig.savefig(OUT/'panels'/f'{name}.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in bounds)),dpi=300)
    fig.savefig(OUT/'panels/c_instance_statistics.png',
        bbox_inches=Bbox.from_extents(*[v/25.4 for v in [.5,.3,179.5,33.5]]),dpi=300)
    plt.close(fig)
    # Compare against the earlier Input version: everything above the chart is exact.
    original=np.asarray(Image.open(STYLE/'figure_F08_final.png'))
    revised=np.asarray(Image.open(OUT/'figure_F08_final.png'))
    assert original.shape==revised.shape
    chart_top=int((h-33.5)*300/25.4)
    assert np.array_equal(original[:chart_top],revised[:chart_top]), 'Upper layout changed'
    assert sha(AUDIT)==sha(OUT/'instance_selection.csv')
    assert all(sha(SOURCE/p)==digest for p,digest in snapshot.items())
    assert all(sha(SOURCE/'renders'/p.name)==sha(p) for p in image_sources.values())
    caption=('Fig. 8. Reference-photo-guided instance localization in an industrial scene containing similar air coolers. '
        'Input: "'+text_a+'". (a) Text-only localization selects the central-aisle cooler; '
        '(b) photo-guided localization selects the intended cooler beside the storage racks. '
        'Both panels and detail windows use registered display camera frame_000183 with identical main crops. '
        'The original M1 reference is frame_000274. Red shows actual saved Gaussian selections; '
        'cyan is an identity cue from the photo-guided selection, not independent ground truth. '
        'Only the user-marked detached contour fragment is omitted from the displayed cue. '
        '(c) Correct/incorrect instance-selection results across 10 confirmed M1 cases: '
        'text-only 4/10 correct (40%) and 6/10 incorrect (60%); photo-guided 10/10 correct (100%). '
        'These are the unchanged saved preliminary AI visual-audit judgments, summarized by M1 case as requested. '
        'The 10 cases cover 9 unique physical targets: M1-04 and M1-09 depict the same cooler from different references '
        'and are counted separately here. All cases belong to one industrial scene; the audit retains saved '
        'description mismatches and is pending independent human adjudication. '
        'No localization, IoU evaluation, generation, Gaussian rendering or optimization was rerun.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    with (OUT/'statistics_summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','correct','incorrect','total_cases','correct_percent','unit'])
        for name,count in zip(['Text-only','Photo-guided (Ours)'],counts):
            writer.writerow([name,count,n-count,n,100*count/n,'M1 case'])
    sources=dict(revision='Restore centered Input style; panel (c) summarizes 10 M1 cases',
        previous_version=str(SOURCE),input_style_source=str(STYLE),
        shared_text_input=dict(path=str(M1_RECORD),sha256=sha(M1_RECORD),field='text',text_a=text_a,
            displayed_text='Input: "'+text_a+'"',verbatim=True,font_pt=8.2,alignment='center',prefix_weight='bold',description_style='italic'),
        candidate_manifest=read(CANDIDATE/'manifest.json'),render_manifest=read(RENDERS/'manifest.json'),
        source_figure_provenance=read(SOURCE/'sources.json'),
        upper_render_files={key:dict(path=str(p),sha256=sha(p)) for key,p in image_sources.items()},
        main_camera='frame_000183.jpg',original_m1_reference_camera='frame_000274.jpg',
        main_crop=MAIN_CROP,target_inset_crop=TARGET_CROP,wrong_instance_crop=WRONG_CROP,
        target_inset_mm=[22.5,22.5],wrong_inset_mm=[21.,21.],contour_cleanup=smoothing,
        figure_mm=[w,h],dpi=300,statistics_source=dict(path=str(AUDIT),sha256=sha(AUDIT),
            unit='confirmed M1 case',case_count=n,unique_physical_target_count=9,
            repeated_target_cases=['M1-04','M1-09'],counts_correct=counts,counts_incorrect=[n-i for i in counts],
            evaluation_status='Preliminary AI visual audit',new_measurement=False),
        statistical_panel_label='(c)',upper_pixels_identical_to_previous_Input_version=True)
    write(OUT/'sources.json',sources)
    write(OUT/'validation.json',dict(previous_version_preserved=True,
        source_render_hashes_unchanged=True,upper_pixels_identical_to_previous_Input_version=True,
        case_statistics_csv_identical_to_saved_audit=True,counts_correct=counts,total_cases=n,
        no_new_localization_or_iou=True,no_generation_or_training=True,only_cached_renders_reused=True))
    write(OUT/'source_hashes.json',dict(directory=str(SOURCE),sha256=snapshot))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - Ten M1 cases</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08</h1><p>Centered Input restored. (c): Text-only 4/10 (40%); Photo-guided 10/10 (100%). Statistics use 10 saved M1 cases.</p>
<p><a href="figure_F08_final.png">300 DPI PNG</a> | <a href="figure_F08_final.pdf">PDF</a> | <a href="caption.txt">Caption</a> | <a href="sources.json">Sources</a></p><img src="figure_F08_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),text_a=text_a,counts_correct=counts,cases=n,upper_pixels_unchanged=True)))

if __name__=='__main__':
    main()
