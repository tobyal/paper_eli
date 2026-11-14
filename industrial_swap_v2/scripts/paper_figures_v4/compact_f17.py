"""Layout-only F17: one enlarged qualitative row and a small mean-only chart.

All data, updated GT, actual splat-render assets and cameras are reused. No
localization, rendering, score recomputation, training or generation is invoked.
"""
import csv, hashlib, html, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
SRC=BASE/'F17_localization_comparison_final'
OUT=BASE/'F17_localization_comparison_compact'
TASK='M1-03_frame_000211'
METHODS=['GaussianEditor','DGE','Ours']
W,H=180.,64.

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main():
    panels=OUT/'panels';code=OUT/'code'
    panels.mkdir(parents=True,exist_ok=True);code.mkdir(exist_ok=True)
    data=read(SRC/'metrics.json');provenance=read(SRC/'sources.json')
    with (SRC/'iou_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        summary={r['method']:r for r in csv.DictReader(f)}
    assert set(summary)==set(METHODS)
    means=[float(summary[m]['mean_heldout_iou']) for m in METHODS]
    assert all(int(summary[m]['valid_equipment_count'])==4 and int(summary[m]['valid_test_view_count'])==6 for m in METHODS)
    scores={r['method']:r['primary_heldout_iou'] for r in data['per_view'] if r['task_id']==TASK}
    sources=[SRC/f for f in ['sources.json','metrics.json','iou_summary.csv','caption.txt','figure_F17_final.png','figure_F17_final.pdf']]
    for method in ['GT']+METHODS:
        sources.append(SRC/'panels'/f'{TASK}_{method if method=="GT" else method+"_projection"}.png')
        if method!='GT':
            sources.extend([SRC/'panels'/f'{TASK}_{method}_gaussian_only_detail.png',SRC/'panels'/f'{TASK}_{method}_gaussian_only_full.png'])
    sources.append(SRC/'ground_truth_masks'/f'{TASK}.png')
    hashes={str(p):sha(p) for p in sources}
    # Preserve the full-precision mean table byte-for-byte. The four-target
    # aggregate still includes M1-05 even though its qualitative row is omitted.
    shutil.copy2(SRC/'iou_summary.csv',OUT/'iou_summary.csv')
    shutil.copy2(SRC/'metrics.json',OUT/'metrics.json')
    shutil.copy2(SRC/'ground_truth_masks'/f'{TASK}.png',panels/'updated_GT_mask.png')
    old_crop=provenance['common_main_crops'][TASK]
    center=(old_crop[0]+old_crop[2])/2
    crop=[round(center-75),old_crop[1],round(center+75),old_crop[3]]
    scale=3;hi_crop=tuple(v*scale for v in crop)
    assert crop==[618,168,768,348]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.5,
        'axes.labelsize':6.5,'xtick.labelsize':6.3,'ytick.labelsize':6.2,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
    def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    fig.text(1.5/W,61/H,'(a) M1-03 | Similar coolers',ha='left',va='center',fontsize=8.5)
    col_x=[1.5,35.,68.5,102.];pw=31.5;py=13.2;ph=37.8;axes=[]
    for i,method in enumerate(['GT']+METHODS):
        title='2D GT' if method=='GT' else method
        fig.text((col_x[i]+pw/2)/W,55/H,title,ha='center',va='center',fontsize=7.8,fontweight='bold')
        path=SRC/'panels'/f'{TASK}_{method if method=="GT" else method+"_projection"}.png'
        im=Image.open(path).convert('RGB')
        cropped=im.crop(hi_crop);cropped.save(panels/f'{method}_main_crop.png',dpi=(300,300))
        shutil.copy2(path,panels/f'{method}_full_projection.png')
        ax=axis(col_x[i],py,pw,ph);axes.append((method,ax));ax.imshow(cropped);ax.axis('off')
        if method=='GT':
            overview=im.resize((938,528),Image.Resampling.LANCZOS)
            ImageDraw.Draw(overview).rectangle(tuple(crop),outline='white',width=2)
            ia=ax.inset_axes([.015,.015,.285,.145]);ia.imshow(overview)
            overview.save(panels/'GT_global_overview.png')
            label='Updated GT mask'
        else:
            detail=SRC/'panels'/f'{TASK}_{method}_gaussian_only_detail.png'
            ia=ax.inset_axes([.015,.015,.285,.44]);ia.set_facecolor('black');ia.imshow(Image.open(detail),aspect='equal')
            ia.text(.5,1.025,'Gaussian-only',transform=ia.transAxes,ha='center',va='bottom',fontsize=4.,color='white',
                bbox=dict(facecolor='black',alpha=1,pad=.7,edgecolor='none'),clip_on=False)
            shutil.copy2(detail,panels/f'{method}_gaussian_only_detail.png')
            shutil.copy2(SRC/'panels'/f'{TASK}_{method}_gaussian_only_full.png',panels/f'{method}_gaussian_only_full.png')
            label=f'IoU = {scores[method]:.3f}'
        ia.set_xticks([]);ia.set_yticks([])
        for spine in ia.spines.values():spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.6)
        fig.text((col_x[i]+pw/2)/W,10.4/H,label,ha='center',va='center',fontsize=7.1)
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.2,label='Selected Gaussians (projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.25,label='GT contour')],loc='center',bbox_to_anchor=(67.5/W,4.4/H),
        ncol=2,frameon=False,fontsize=6.3,handlelength=1.7,columnspacing=1.5)
    # Mean-only chart: no per-device/per-view points, error bars or extra series.
    ax=axis(145,14.5,33.5,35.5)
    colors=['#be7d34','#697b8f','#25799a']
    ax.bar(range(3),means,.55,color=colors,zorder=3)
    for j,val in enumerate(means):ax.text(j,val+.035,f'{val:.3f}',ha='center',va='bottom',fontsize=7.3,fontweight='bold')
    ax.set_xlim(-.55,2.55);ax.set_ylim(0,1.04);ax.set_yticks([0,.5,1]);ax.set_ylabel('IoU',labelpad=2)
    ax.set_xticks(range(3),['Gaussian\nEditor','DGE','Ours']);ax.tick_params(axis='x',length=2,pad=3,labelsize=6.1)
    ax.tick_params(axis='y',length=2,pad=2);ax.grid(axis='y',color='#e9e9e9',zorder=0)
    fig.text(161.75/W,57.5/H,'Mean Held-out\nLocalization IoU',ha='center',va='center',fontsize=7.5,linespacing=1.15)
    fig.text(161.75/W,52/H,'4 devices / 6 test views',ha='center',va='center',fontsize=5.9,color='#444')
    assert len(ax.patches)==3 and len(ax.collections)==0
    fig.savefig(OUT/'figure_F17_compact.png',dpi=300)
    fig.savefig(OUT/'figure_F17_compact.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for name,a in axes+[('mean_iou',ax)]:
        box=a.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.015,1.03)
        fig.savefig(panels/f'{name}_panel.png',bbox_inches=box,dpi=300)
    plt.close(fig)
    caption=('Fig. 17. Gaussian-based localization of industrial equipment. '
        '(a) The M1-03 similar-cooler example compares GaussianEditor, DGE and Ours using the same registered RGB photograph, camera and shared crop. '
        'Cyan outlines the latest user-provided 2D GT mask; red shows the cached continuous projection of the actual selected Gaussians. '
        'Black-background Gaussian-only insets show same-camera local detail. Full selected-set renders, including off-crop false positives, remain available separately. '
        'The display-only opacity multiplier (0.12) and overlay gain (0.80) are unchanged and are not used for IoU. '
        'Right: mean held-out localization IoU across the same four valid physical targets and six eligible test views; views are averaged within each target before taking the unweighted target mean. '
        f'Mean IoU is {means[0]:.4f}, {means[1]:.4f} and {means[2]:.4f} for GaussianEditor, DGE and Ours, respectively. '
        'Omitting the M1-05 qualitative row does not remove that target from the aggregate. '
        'Two evaluation masks are user-provided updates and four remain unreviewed AI drafts, so the aggregate remains a preliminary mixed-annotation result in one scene. '
        'Missing targets are excluded rather than assigned zero. Scores use the original full-frame 938 x 528 prediction masks at threshold 0.05; all camera exclusions and experimental results are unchanged. '
        'This revision changes only figure layout and display framing; no localization, model generation or Gaussian rendering is rerun.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    assert sha(OUT/'iou_summary.csv')==sha(SRC/'iou_summary.csv')
    assert hashes=={str(p):sha(p) for p in sources}
    current_gt=provenance['ground_truth'][TASK]
    write(OUT/'sources.json',dict(source_directory=str(SRC),revision_scope='Layout-only, one qualitative case and right-hand mean-only bar chart',
        retained_qualitative_case=TASK,omitted_qualitative_case='M1-05_frame_000228',
        omitted_qualitative_case_remains_in_mean=True,previous_source_record=provenance,
        shared_main_crop=crop,previous_main_crop=old_crop,crop_transform='Same vertical range; symmetric horizontal tightening to 150 x 180 for all four columns',
        camera=current_gt['camera'],updated_GT_sha256=current_gt['sha256'],
        figure_mm=[W,H],dpi=300,vector_pdf_text=True,left_panel_width_mm=pw,left_panel_height_mm=ph,
        chart_position='right',chart_width_mm=33.5,bar_count=3,scatter_count=0,
        mean_values=dict(zip(METHODS,means)),qualitative_IoU=scores,
        valid_equipment_count=4,valid_test_view_count=6,
        original_images_and_statistics_unchanged=True,summary_CSV_byte_identical=True,
        source_hashes=hashes,files_reused_without_new_rendering=True))
    write(OUT/'validation.json',dict(source_hashes_unchanged=True,CSV_copied_byte_for_byte=True,
        single_qualitative_case=True,mean_bar_count=3,per_device_or_per_view_scatter_count=0,
        main_crop_identical_for_all_methods=True,original_final_figure_preserved=True,
        image_pixels=list(Image.open(OUT/'figure_F17_compact.png').size)))
    shutil.copy2(__file__,code/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17_compact.pdf'),str(OUT/'pdf_proof')],check=True)
    page=f'''<!doctype html><meta charset="utf-8"><title>F17 compact</title><style>body{{font:16px/1.6 system-ui;max-width:1550px;margin:auto;padding:26px}}img{{max-width:100%}}a{{color:#155da1}}</style>
<h1>F17 · 单案例紧凑版</h1><p>左侧仅保留 M1-03；右侧为三个方法的均值柱，不含散点。均值仍覆盖 4 个设备、6 个视角，M1-05 只从定性展示中删除。</p>
<p><a href="figure_F17_compact.png">300 DPI PNG</a> · <a href="figure_F17_compact.pdf">PDF</a> · <a href="iou_summary.csv">原值 CSV</a> · <a href="caption.txt">英文图注</a> · <a href="sources.json">来源与相机</a> · <a href="../F17_localization_comparison_final/index.html">上一版</a></p>
<img src="figure_F17_compact.png" alt="F17 compact"><p>{html.escape(caption)}</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('Compact F17 exported: 180 x 64 mm, one qualitative case, three mean-only bars; previous assets and CSV unchanged.',flush=True)

if __name__=='__main__':main()
