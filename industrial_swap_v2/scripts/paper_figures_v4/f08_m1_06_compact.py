"""Final F08 typography: left-aligned shared query and panel (c), evidence unchanged."""
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
OUT=BASE/'F08_final_M1_06_compact'
SOURCE=BASE/'F08_final_M1_06_adjusted'
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
    shutil.copy2(SOURCE/'instance_selection.csv',OUT/'instance_selection.csv')
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
    counts=[sum(int(r[k]) for r in rows) for k in ['text_only_correct','photo_guided_correct']]
    assert len(rows)==9 and counts==[3,9]
    # Reuse the earlier real splat renders with 60% red color mixing, exactly.
    image_sources={key:RENDERS/(key+'.png') for key in ['a_main','b_main','wrong_instance']}
    images={key:Image.open(p).convert('RGB') for key,p in image_sources.items()}

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    # Add exactly 72 pixels at 300 DPI above the unchanged previous layout.
    # This preserves the pixel grid for the accepted lower chart.
    w,h=180.,108.+24*25.4/300.  # Remove 48 pixels of top whitespace at 300 DPI.
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
        TextArea('Shared text query:',textprops=dict(fontsize=7.2,fontweight='normal',color='#222222')),
        TextArea('"'+text_a+'"',textprops=dict(fontsize=7.2,fontstyle='italic',color='#222222'))],
        align='center',pad=0,sep=4)
    input_anchor=AnchoredOffsetbox(loc='center left',child=input_line,pad=0,frameon=False,
        bbox_to_anchor=(1.5/w,(h-2.2)/h),bbox_transform=fig.transFigure,borderpad=0)
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

    # Accepted panel (d): exact same sizes, positions, text, colors and data.
    text(1.5,31,'(c) Correct vs. Incorrect Instance Selection',ha='left',va='center',fontsize=8.1)
    correct='#196f86';incorrect='#d9a277'
    fig.legend(handles=[Patch(facecolor=correct,label='Correct'),Patch(facecolor=incorrect,label='Incorrect')],
        loc='center',bbox_to_anchor=(155/w,31/h),ncol=2,frameon=False,fontsize=6.1,
        handlelength=1.2,handletextpad=.4,columnspacing=1.2)
    ax=axis(36,11,139,16.5)
    values=np.array(counts)/9.
    ax.barh([1,0],values,height=.62,color=correct,zorder=3)
    ax.barh([1,0],1-values,left=values,height=.62,color=incorrect,zorder=3)
    ax.text(1/6,1,'3/9 (33.3%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(2/3,1,'6/9 (66.7%)',ha='center',va='center',fontsize=7.1,color='#3f2b20',fontweight='bold')
    ax.text(.5,0,'9/9 (100%)',ha='center',va='center',fontsize=7.1,color='white',fontweight='bold')
    ax.text(.985,0,'Incorrect: 0/9 (0%)',ha='right',va='center',fontsize=5.8,color='white')
    ax.set_yticks([1,0],['Text-only','Photo-guided (Ours)']);ax.tick_params(axis='y',length=0,labelsize=7.1,pad=5)
    ax.set_ylim(-.55,1.55);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(axis='x',labelsize=6.1,length=2,pad=2)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#777777')
    ax.grid(axis='x',color='#ededed',zorder=0)
    text(1.5,2.3,'9 unique equipment targets',ha='left',va='center',fontsize=5.9,color='#555555')
    text(178.5,2.3,'Preliminary AI visual audit',ha='right',va='center',fontsize=5.9,color='#555555')
    fig.savefig(OUT/'figure_F08_final.png',dpi=300);fig.savefig(OUT/'figure_F08_final.pdf')
    for col,name in enumerate(['a_text_only','b_photo_guided']):
        bounds=[.5+90*col,40,89.5+90*col,105.8]
        fig.savefig(OUT/'panels'/f'{name}.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in bounds)),dpi=300)
    fig.savefig(OUT/'panels/c_instance_statistics.png',
        bbox_inches=Bbox.from_extents(*[v/25.4 for v in [.5,.3,179.5,33.5]]),dpi=300)
    plt.close(fig)
    original=np.asarray(Image.open(SOURCE/'figure_F08_final.png'))
    revised=np.asarray(Image.open(OUT/'figure_F08_final.png'))
    assert revised.shape[1:]==original.shape[1:]
    removed_rows=original.shape[0]-revised.shape[0]
    assert removed_rows==48
    aligned=original[removed_rows:]
    # Preserve all pixels below the shared-query band except the statistics title.
    # The latter changes its panel letter from (d) to (c), including glyph advance.
    unchanged_region=np.ones(revised.shape[:2],dtype=bool)
    y_mm=h-np.arange(revised.shape[0])*25.4/300.
    x_mm=np.arange(revised.shape[1])*25.4/300.
    unchanged_region[y_mm>106.,:]=False
    title_zone=((y_mm>=28.5)&(y_mm<=33.5))[:,None] & (x_mm<115.)[None,:]
    unchanged_region[title_zone]=False
    pixel_delta=np.max(np.abs(aligned.astype(np.int16)-revised.astype(np.int16)),axis=2)
    nonedited_max_difference=int(pixel_delta[unchanged_region].max())
    nonedited_rounding_pixels=int(((pixel_delta>0)&unchanged_region).sum())
    # Matplotlib can round a few resampled edge pixels differently after a
    # page-height change; source images, geometry and layout stay identical.
    assert nonedited_max_difference<=1 and nonedited_rounding_pixels<100,'Unrequested figure content changed'
    start=int(np.floor((h-28.5)*300/25.4))
    assert np.array_equal(aligned[start:],revised[start:]),'Chart bars or values changed'
    assert sha(OUT/'instance_selection.csv')==sha(SOURCE/'instance_selection.csv')
    assert all(sha(SOURCE/p)==digest for p,digest in snapshot.items())
    caption=('Fig. 8. Reference-photo-guided instance localization in an industrial scene containing similar air coolers. '
        'The shared instruction, "'+text_a+'", is read verbatim from the confirmed M1-06 record. '
        '(a) Text-only localization selects the central-aisle cooler, whereas (b) photo-guided localization '
        'selects the intended cooler beside the storage racks. Both panels and all detail windows use '
        'the same registered display camera (frame_000183) and the two main panels have identical crops. '
        'The original M1 assistance used the registered reference frame_000274 and its user-selected SAM region; '
        'the display camera is not a new localization input. Cyan outlines the visible projection of the saved '
        'photo-guided target Gaussian selection and is an identity cue, not independent ground truth. '
        'The original irregular contour is retained except for one detached line fragment '
        'explicitly marked by the user, which is omitted only from the displayed identity cue '
        '(12 pixels at the native display-mask resolution). No smoothing is applied. '
        'The original mask, red Gaussian rendering and evaluation data remain unchanged. '
        'Red highlights the actual saved selections through Gaussian rasterization using a 60% red color '
        'mixture, with geometry, opacity, camera parameters and selected indices unchanged. '
        'Foreground occlusions and the actual Gaussian boundary spill remain visible in the red rendering. '
        'The Intended Target and Correct Target windows occupy the same upper-right location '
        'and size in (a) and (b). The smaller Wrong Instance window is at the lower right in (a). '
        'All three detail headers use black text on white. The smaller shared description is '
        'left-aligned with panel (a) above the two panel titles and prefixed Shared text query. '
        '(c) The previously accepted horizontal correct/incorrect instance-selection chart is unchanged: '
        'text-only 3/9 correct and 6/9 incorrect; photo-guided 9/9 correct and 0/9 incorrect. '
        'M1-04 and M1-09 are grouped as one physical cooler. These judgments remain a preliminary '
        'AI visual audit pending independent human adjudication; all nine targets belong to one industrial '
        'scene. The audit cohort retains some saved description mismatches, although this selected '
        'qualitative example uses a category-compatible cooler description. No localization, IoU '
        'evaluation, generation, rendering or optimization is rerun; existing cached renderings are reused.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    source_record=dict(path=str(M1_RECORD),sha256=sha(M1_RECORD),field='text',
        m1_case_id='a0ee4103b79a4fcb8911ffaecb1720aa',display_case='M1-06',
        text_a=text_a,displayed_text='Shared text query: "'+text_a+'"',verbatim=True)
    write(OUT/'sources.json',dict(revision='Smaller shared text query aligned to left edge of panel (a); less top whitespace; statistics renumbered (c)',
        previous_version=str(SOURCE),previous_sources=read(SOURCE/'sources.json'),
        shared_text_input=source_record,candidate_manifest=read(CANDIDATE/'manifest.json'),
        candidate_approved_by_user=True,render_manifest=read(RENDERS/'manifest.json'),
        cyan_contour_source='Original saved photo-assisted projection except the single user-marked detached fragment; no smoothing; not independent GT',
        contour_cleanup=smoothing,
        inset_positions=dict(a_intended='upper right',a_wrong='lower right',b_correct='upper right'),
        previous_main_crop=MAIN_CROP,main_crop_adjustment='None; both scene panels and all inset coordinates unchanged',
        upper_render_files={key:dict(path=str(p),sha256=sha(p)) for key,p in image_sources.items()},
        figure_mm=[w,h],dpi=300,main_panel_mm=[pw,ph],main_crop=MAIN_CROP,
        target_inset_crop=TARGET_CROP,wrong_instance_crop=WRONG_CROP,
        target_inset_mm=[target_inset_size]*2,wrong_inset_mm=[wrong_inset_size]*2,
        prior_target_inset_mm=[18.,18.],prior_wrong_inset_mm=[27.3,27.3],
        wrong_inset_linear_scale=21./27.3,target_inset_linear_scale=22.5/18.,
        inset_header_height_mm=inset_header_height,inset_header_font_pt=5.8,
        inset_header_background='white',inset_header_text='black',shared_input_prefix='Shared text query:',
        shared_query_font_pt=7.2,shared_query_alignment='left edge of panel (a)',
        shared_query_anchor_mm=[1.5,h-2.2],top_whitespace_reduction_mm=48*25.4/300.,statistical_panel_label='(c)',
        main_camera='frame_000183.jpg',wrong_instance_camera='frame_000183.jpg',
        original_m1_reference_camera='frame_000274.jpg',
        upper_red_style='Unmodified actual Gaussian render: 0.60 red color mixture, original opacity and geometry',
        lower_chart_data_and_bars_pixel_identical=True,statistical_title_only_renumbered=True,
        lower_chart_output_pixel_row=start,nonedited_image_content_preserved=True,
        nonedited_max_channel_difference=nonedited_max_difference,nonedited_rounding_pixel_count=nonedited_rounding_pixels,
        lower_chart_source=str(SOURCE/'panels/d_instance_statistics.png'),
        lower_chart_output=str(OUT/'panels/c_instance_statistics.png'),
        lower_chart_sha256=sha(OUT/'panels/c_instance_statistics.png'),CSV_byte_identical=True,
        counts_correct=counts,counts_incorrect=[9-n for n in counts],unique_targets=9))
    write(OUT/'validation.json',dict(previous_version_preserved=True,original_hashes_unchanged=True,
        lower_chart_data_and_bars_pixel_identical=True,statistical_title_only_renumbered=True,
        unchanged_content_max_channel_difference=nonedited_max_difference,
        unchanged_content_rounding_pixel_count=nonedited_rounding_pixels,
        all_source_image_files_unchanged=True,
        statistics_CSV_byte_identical=True,two_equal_main_panels=True,target_and_correct_insets_same_size=True,
        wrong_inset_shrunk_from_mm=27.3,wrong_inset_mm=21.,target_insets_mm=22.5,
        target_insets_aligned=True,only_user_circled_fragment_removed=True,no_contour_smoothing=True,shared_text_verbatim_from_m1=True,
        no_new_localization_or_iou=True,no_generation_or_training=True,
        only_cached_renders_reused=True,display_contour_cleanup_only=True,
        raw_mask_and_cached_render_files_byte_identical=True))
    write(OUT/'source_hashes.json',dict(directory=str(SOURCE),sha256=snapshot))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - Final shared text query layout</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08 · 最终紧凑版式</h1><p>Shared text query 左对齐至 (a) 图片边缘，字号调整为 7.2 pt，描述保持斜体双引号，减少顶部留白。统计图编号改为 (c)。定位图、三个小窗、红色高斯、青色轮廓与统计数据均保留。</p>
<p><a href="figure_F08_final.png">300 DPI PNG</a> · <a href="figure_F08_final.pdf">PDF</a> · <a href="panels/">独立子图</a> · <a href="instance_selection.csv">原值 CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F08_final_M1_06_adjusted/index.html">上一版</a></p><img src="figure_F08_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('F08 optimized. Shared Text A:',text_a,flush=True)
    print('All image panels and chart bars unchanged; only query typography and statistical title differ.',flush=True)

if __name__=='__main__':
    main()
