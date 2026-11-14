"""Restore F08's two red Gaussian views; retain the accepted lower chart exactly."""
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
OUT=BASE/'F08_final_relayout'
BACKUP=BASE/'F08_final_relayout_backup_three_panels'
OLD=BASE/'F08_localization_final'
RENDERS=OLD/'renders_revised'
MAIN_CROP=[192,44,896,528]
TARGET_CROP=[602,315,752,465]
WRONG_CROP=[705,235,855,385]
SCALE=3

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if not BACKUP.exists():
        assert (OUT/'panels/d_instance_statistics.png').is_file()
        assert OUT.resolve().parent==BACKUP.resolve().parent==BASE.resolve()
        OUT.rename(BACKUP)
    for folder in ['panels','inputs','code','renders']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    snapshot={str(p.relative_to(BACKUP)):sha(p) for p in BACKUP.rglob('*') if p.is_file()}
    shutil.copy2(BACKUP/'instance_selection.csv',OUT/'instance_selection.csv')
    shutil.copytree(BACKUP/'inputs',OUT/'inputs',dirs_exist_ok=True)
    mask=np.asarray(Image.open(OUT/'inputs/intended_target_mask.png'))>127
    with (OUT/'instance_selection.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    counts=[sum(int(r[k]) for r in rows) for k in ['text_only_correct','photo_guided_correct']]
    assert len(rows)==9 and counts==[3,9]
    # Reuse the earlier real splat renders with 60% red color mixing, exactly.
    image_sources={key:RENDERS/(key+'.png') for key in ['a_main','b_main','wrong_instance']}
    images={key:Image.open(p).convert('RGB') for key,p in image_sources.items()}
    for key,p in image_sources.items():shutil.copy2(p,OUT/'renders'/p.name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    w,h=180.,108.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    def show(ax,key,crop,cyan):
        ax.imshow(images[key],extent=[-.5,937.5,527.5,-.5],interpolation='lanczos')
        if cyan:ax.contour(mask,levels=[.5],colors=['#00cfe7'],linewidths=.95,antialiased=True)
        ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5)
        ax.set_xticks([]);ax.set_yticks([])
    pw=87.;ph=pw*(MAIN_CROP[3]-MAIN_CROP[1])/(MAIN_CROP[2]-MAIN_CROP[0])
    inset_size=21.
    for col,(key,title) in enumerate([('a_main','(a) Text-only Localization'),('b_main','(b) Photo-guided Localization')]):
        x=1.5+90*col;y=41.
        ax=axis(x,y,pw,ph);show(ax,key,MAIN_CROP,True)
        for spine in ax.spines.values():spine.set_visible(False)
        text(x,y+ph+3.1,title,ha='left',va='center',fontsize=8.4)
        insets=[(x+pw-inset_size-1.2,y+ph-inset_size-4.2,key,TARGET_CROP,
                 'Intended target' if col==0 else 'Correct target',True)]
        if col==0:insets.append((x+1.2,y+1.2,'wrong_instance',WRONG_CROP,'Wrong instance',False))
        for zx,zy,zkey,box,label,cyan in insets:
            ia=axis(zx,zy,inset_size,inset_size);show(ia,zkey,box,cyan)
            for spine in ia.spines.values():
                spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.8)
            # A separate uniform header leaves the entire equipment crop visible.
            ia.add_patch(Rectangle((0,1),1,.14,transform=ia.transAxes,clip_on=False,
                facecolor='#171717',alpha=.9,edgecolor='white',linewidth=.8))
            ia.text(.035,1.07,label,transform=ia.transAxes,va='center',ha='left',
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
    original=np.asarray(Image.open(BACKUP/'figure_F08_final.png'))
    revised=np.asarray(Image.open(OUT/'figure_F08_final.png'))
    assert original.shape==revised.shape
    start=round((h-33.5)/h*original.shape[0])
    assert np.array_equal(original[start:],revised[start:]),'Accepted lower panel changed'
    shutil.copy2(BACKUP/'panels/d_instance_statistics.png',OUT/'panels/d_instance_statistics.png')
    assert sha(OUT/'instance_selection.csv')==sha(BACKUP/'instance_selection.csv')
    assert all(sha(BACKUP/p)==digest for p,digest in snapshot.items())
    caption=('Fig. 8. Reference-photo-guided instance localization in an industrial scene. '
        '(a) Text-only and (b) photo-guided localization use the same saved registered main camera '
        '(frame_000342) and shared scene crop. Cyan outlines the user-designated M1-10 target. '
        'Red highlights the actual selected Gaussians using the earlier 60% red color mixture; '
        'the stored Gaussian geometry, opacity and depth compositing remain unchanged. '
        'The upper-right detail windows show the intended target and correct localization; the lower-left '
        'window in (a) shows the text-only wrong instance using its actual saved camera (frame_000015). '
        'All three detail windows use the same size and border/label style. '
        '(d) The previously accepted horizontal correct/incorrect instance-selection chart is unchanged, '
        'including its pixels, labels and data: text-only 3/9 correct and 6/9 incorrect, photo-guided '
        '9/9 correct and 0/9 incorrect. M1-04 and M1-09 are grouped as one physical cooler. '
        'The judgments remain a preliminary AI visual audit, pending independent human adjudication. '
        'Some saved descriptions, including M1-10, imperfectly describe the user-selected equipment, '
        'so description mismatch is a confound in addition to instance ambiguity. All nine physical '
        'targets belong to one industrial scene. No localization, rendering, generation or optimization '
        'is rerun; the upper panels reuse existing actual Gaussian-rendered images.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    write(OUT/'sources.json',dict(revision='Two red Gaussian scene panels with aligned insets; accepted panel (d) preserved',
        previous_version_backup=str(BACKUP),previous_sources=read(BACKUP/'sources.json'),
        upper_render_manifest=read(RENDERS/'manifest.json'),
        upper_render_files={key:dict(path=str(p),sha256=sha(p)) for key,p in image_sources.items()},
        figure_mm=[w,h],dpi=300,main_panel_mm=[pw,ph],main_crop=MAIN_CROP,
        target_inset_crop=TARGET_CROP,wrong_instance_crop=WRONG_CROP,inset_mm=[inset_size,inset_size],
        main_camera='frame_000342.jpg',wrong_instance_camera='frame_000015.jpg',
        upper_red_style='Prior actual Gaussian render: 0.60 red color mixture, original opacity and geometry',
        lower_chart_pixel_identical=True,lower_chart_check_from_pixel_row=start,
        lower_chart_source=str(BACKUP/'panels/d_instance_statistics.png'),
        lower_chart_sha256=sha(OUT/'panels/d_instance_statistics.png'),CSV_byte_identical=True,
        counts_correct=counts,counts_incorrect=[9-n for n in counts],unique_targets=9))
    write(OUT/'validation.json',dict(previous_version_preserved=True,original_hashes_unchanged=True,
        lower_chart_pixel_identical=True,lower_chart_check_from_pixel_row=start,
        lower_chart_file_byte_identical=True,statistics_CSV_byte_identical=True,
        two_equal_main_panels=True,three_equal_insets=True,no_experiments_or_new_rendering=True))
    write(OUT/'backup_manifest.json',dict(directory=str(BACKUP),sha256=snapshot))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F08 - Two scene panels</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F08 · 两张主图与原统计图</h1><p>上方恢复两张定位主图和醒目红色；三个小窗统一为 21 mm 方形。下方 (d) 横向堆叠统计图已核验像素不变。</p>
<p><a href="figure_F08_final.png">300 DPI PNG</a> · <a href="figure_F08_final.pdf">PDF</a> · <a href="panels/">独立子图</a> · <a href="instance_selection.csv">原值 CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F08_final_relayout_backup_three_panels/index.html">上一版</a></p><img src="figure_F08_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('F08 upper panels restored; accepted lower chart is pixel-identical from row',start,flush=True)

if __name__=='__main__':main()
