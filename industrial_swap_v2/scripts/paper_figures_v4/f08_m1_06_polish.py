"""Display-only contour cleanup and inset swap; retain all M1-06 evidence."""
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
OUT=BASE/'F08_final_M1_06_polished'
SOURCE=BASE/'F08_final_M1_06'
CANDIDATE=BASE/'F08_candidate_selection/candidate_02'
RENDERS=OUT/'renders'
M1_RECORD=ROOT/'industrial_swap_v2/webapp/data/m1_assist/cases/a0ee4103b79a4fcb8911ffaecb1720aa/job.json'
MAIN_CROP=[0,0,680,468]  # Upward framing keeps the larger upper-right inset clear of the cooler.
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
    from scipy import ndimage as ndi
    components,n=ndi.label(mask)
    sizes=np.bincount(components.ravel());sizes[0]=0
    largest=components==sizes.argmax()
    yy,xx=np.mgrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    clean=ndi.binary_opening(largest,structure=disk)
    clean=ndi.binary_fill_holes(ndi.binary_closing(clean,structure=disk))
    components,n=ndi.label(clean);sizes=np.bincount(components.ravel());sizes[0]=0
    clean=components==sizes.argmax()
    display_contour=ndi.gaussian_filter(clean.astype(np.float32),sigma=1.4)
    Image.fromarray(((display_contour>.5)*255).astype(np.uint8)).save(OUT/'inputs/intended_contour_display_only.png')
    np.save(OUT/'inputs/intended_contour_display_field.npy',display_contour)
    smoothing=dict(display_only=True,original_mask_unchanged=True,largest_component=True,
        morphological_open_close_radius_pixels=2,gaussian_sigma_pixels=1.4,
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
    w,h=180.,108.+72*25.4/300.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    def show(ax,key,crop,cyan):
        ax.imshow(images[key],extent=[-.5,937.5,527.5,-.5],interpolation='lanczos')
        if cyan:ax.contour(display_contour,levels=[.5],colors=['#00cfe7'],linewidths=.95,antialiased=True)
        ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5)
        ax.set_xticks([]);ax.set_yticks([])
    pw=87.;ph=pw*(MAIN_CROP[3]-MAIN_CROP[1])/(MAIN_CROP[2]-MAIN_CROP[0])
    target_inset_size=18.
    wrong_inset_size=21.*1.3
    inset_header_height=2.8
    text(w/2,h-3.0,'"'+text_a+'"',ha='center',va='center',
         fontsize=8.2,fontstyle='italic',color='#222222')
    for col,(key,title) in enumerate([('a_main','(a) Text-only Localization'),('b_main','(b) Photo-guided Localization')]):
        x=1.5+90*col;y=41.
        ax=axis(x,y,pw,ph);show(ax,key,MAIN_CROP,True)
        for spine in ax.spines.values():spine.set_visible(False)
        text(x,y+ph+3.1,title,ha='left',va='center',fontsize=8.4)
        if col==0:
            insets=[(x+1.2,y+1.2,target_inset_size,key,TARGET_CROP,'Intended target',True),
                (x+pw-wrong_inset_size-1.2,y+ph-wrong_inset_size-inset_header_height-1.2,
                 wrong_inset_size,'wrong_instance',WRONG_CROP,'Wrong instance',False)]
        else:
            insets=[(x+pw-target_inset_size-1.2,y+ph-target_inset_size-inset_header_height-1.2,
                target_inset_size,key,TARGET_CROP,'Correct target',True)]
        for zx,zy,size,zkey,box,label,cyan in insets:
            ia=axis(zx,zy,size,size);show(ia,zkey,box,cyan)
            for spine in ia.spines.values():
                spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.8)
            # Fixed physical header height and font size across all inset sizes.
            header_fraction=inset_header_height/size
            ia.add_patch(Rectangle((0,1),1,header_fraction,transform=ia.transAxes,clip_on=False,
                facecolor='#171717',alpha=.9,edgecolor='white',linewidth=.8))
            ia.text(.035,1+header_fraction/2,label,transform=ia.transAxes,va='center',ha='left',
                fontsize=5.6,color='white',clip_on=False)
        images[key].crop(tuple(v*SCALE for v in MAIN_CROP)).save(OUT/'panels'/f'{key}_scene.png',dpi=(300,300))
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.8,label='Selected Gaussians'),
        Line2D([0],[0],color='#00cfe7',lw=1.5,label='Intended target')],loc='center',bbox_to_anchor=(.5,35.8/h),
        ncol=2,frameon=False,fontsize=6.7,columnspacing=1.5,handlelength=1.6)

    # Accepted panel (d): exact same sizes, positions, text, colors and data.
    text(1.5,31,'(d) Correct vs. Incorrect Instance Selection',ha='left',va='center',fontsize=8.1)
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
        bounds=[.5+90*col,40,89.5+90*col,107.]
        fig.savefig(OUT/'panels'/f'{name}.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in bounds)),dpi=300)
    plt.close(fig)
    original=np.asarray(Image.open(SOURCE/'figure_F08_final.png'))
    revised=np.asarray(Image.open(OUT/'figure_F08_final.png'))
    assert revised.shape[1:]==original.shape[1:]
    extra_rows=revised.shape[0]-original.shape[0]
    assert extra_rows==0
    start=round((108.-33.5)/108.*1275)+72
    assert np.array_equal(original[start:],revised[start:]),'Accepted lower panel changed'
    shutil.copy2(SOURCE/'panels/d_instance_statistics.png',OUT/'panels/d_instance_statistics.png')
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
        'Its display contour is lightly regularized to remove isolated speckles and jagged edges; '
        'the original mask, red Gaussian rendering and evaluation data are retained unchanged. '
        'Red highlights the actual saved selections through Gaussian rasterization using a 60% red color '
        'mixture, with geometry, opacity, camera parameters and selected indices unchanged. '
        'Foreground occlusions and the actual Gaussian boundary spill remain visible in the red rendering. '
        'In (a), the intended-target window is at the lower left and the larger wrong-instance window '
        'is at the upper right. The correct-target window in (b) remains at the upper right. '
        '(d) The previously accepted horizontal correct/incorrect instance-selection chart is unchanged: '
        'text-only 3/9 correct and 6/9 incorrect; photo-guided 9/9 correct and 0/9 incorrect. '
        'M1-04 and M1-09 are grouped as one physical cooler. These judgments remain a preliminary '
        'AI visual audit pending independent human adjudication; all nine targets belong to one industrial '
        'scene. The audit cohort retains some saved description mismatches, although this selected '
        'qualitative example uses a category-compatible cooler description. No localization, IoU '
        'evaluation, generation, rendering or optimization is rerun; existing cached renderings are reused.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    source_record=dict(path=str(M1_RECORD),sha256=sha(M1_RECORD),field='text',
        m1_case_id='a0ee4103b79a4fcb8911ffaecb1720aa',display_case='M1-06',
        text_a=text_a,displayed_text='"'+text_a+'"',verbatim=True)
    write(OUT/'sources.json',dict(revision='Lightly smoothed cyan display contour; intended and wrong-instance windows swapped in (a); panel (d) unchanged',
        previous_version=str(SOURCE),previous_sources=read(SOURCE/'sources.json'),
        shared_text_input=source_record,candidate_manifest=read(CANDIDATE/'manifest.json'),
        candidate_approved_by_user=True,render_manifest=read(RENDERS/'manifest.json'),
        cyan_contour_source='Regularized display boundary derived from saved photo-assisted projection; not independent GT',
        contour_cleanup=smoothing,
        inset_positions=dict(a_intended='lower left',a_wrong='upper right',b_correct='upper right'),
        previous_main_crop=[0,44,704,528],main_crop_adjustment='Shifted upward 44 pixels and tightened from 704x484 to 680x468 so the enlarged upper-right inset clears the wrong cooler',
        upper_render_files={key:dict(path=str(p),sha256=sha(p)) for key,p in image_sources.items()},
        figure_mm=[w,h],dpi=300,main_panel_mm=[pw,ph],main_crop=MAIN_CROP,
        target_inset_crop=TARGET_CROP,wrong_instance_crop=WRONG_CROP,
        target_inset_mm=[target_inset_size]*2,wrong_inset_mm=[wrong_inset_size]*2,
        prior_inset_mm=[21.,21.],wrong_inset_linear_scale=1.3,target_inset_linear_scale=18./21.,
        inset_header_height_mm=inset_header_height,inset_header_font_pt=5.6,
        main_camera='frame_000183.jpg',wrong_instance_camera='frame_000183.jpg',
        original_m1_reference_camera='frame_000274.jpg',
        upper_red_style='Unmodified actual Gaussian render: 0.60 red color mixture, original opacity and geometry',
        lower_chart_pixel_identical=True,lower_chart_source_pixel_row=start,
        lower_chart_output_pixel_row=start+extra_rows,
        lower_chart_source=str(SOURCE/'panels/d_instance_statistics.png'),
        lower_chart_sha256=sha(OUT/'panels/d_instance_statistics.png'),CSV_byte_identical=True,
        counts_correct=counts,counts_incorrect=[9-n for n in counts],unique_targets=9))
    write(OUT/'validation.json',dict(previous_version_preserved=True,original_hashes_unchanged=True,
        lower_chart_pixel_identical=True,lower_chart_file_byte_identical=True,
        statistics_CSV_byte_identical=True,two_equal_main_panels=True,target_and_correct_insets_same_size=True,
        wrong_inset_enlargement='30% in width and height',shared_text_verbatim_from_m1=True,
        no_new_localization_or_iou=True,no_generation_or_training=True,
        only_cached_renders_reused=True,display_contour_cleanup_only=True,
        raw_mask_and_cached_render_files_byte_identical=True))
    write(OUT/'source_hashes.json',dict(directory=str(SOURCE),sha256=snapshot))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - M1-06 refined display</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08 · M1-06 冷风机实例定位</h1><p>青色提示轮廓已做轻度去毛边；(a) 的 Intended target 移到左下，Wrong instance 移到右上并保留原来的放大尺寸。两张主图使用同一微调后的裁剪窗口，避免小窗遮住设备。高斯渲染、原始 Mask 和下方统计图均保持不变。</p>
<p><a href="figure_F08_final.png">300 DPI PNG</a> · <a href="figure_F08_final.pdf">PDF</a> · <a href="panels/">独立子图</a> · <a href="instance_selection.csv">原值 CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F08_final_M1_06/index.html">上一版</a></p><img src="figure_F08_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('F08 optimized. Shared Text A:',text_a,flush=True)
    print('Lower chart verified pixel-identical; old files unchanged.',flush=True)

if __name__=='__main__':
    main()
