"""Publication F08 from immutable real Gaussian projections and audited metrics.

No model calls. Missing held-out annotations remain visibly NA, never zero.
"""
from pathlib import Path
import argparse, json, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F08_localization_revised'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')

def main(output_dir=None,source_dir=None):
    global D
    source_dir=Path(source_dir or D)
    D=Path(output_dir or D)
    data=read(D/'metrics.json');rows=data['per_case'];summary=data['summary']
    draft=summary.get('evaluation_mode')=='ai_draft'
    supplied=summary.get('evaluation_mode')=='user_supplied_results'
    suffix='reported_iou' if supplied else ('ai_draft_iou' if draft else 'heldout_iou')
    count_key='number_ai_draft_evaluation_views' if draft else 'number_valid_evaluation_views'
    record=next(x for x in read(D/'case_index.json')['cases'] if x['catalog_number']==10)
    source=source_dir/'cases'/record['m1_case_id'];panels=D/'panels';panels.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    fig=plt.figure(figsize=(180/25.4,129/25.4),facecolor='white')
    # Same crop for the two actual reference-camera renders. No image repair.
    mask=np.asarray(Image.open(source/'reference_mask.png'))>127
    # Preserve the complete registered view to match the accepted Case10 evidence.
    axes=[]
    for i,(method,label) in enumerate([('text_only','Text-only Localization'),('photo_assisted','Photo-assisted Localization')]):
        ax=fig.add_axes([.018+i*.495,.535,.47,.405]);axes.append(ax)
        im=Image.open(source/'view_0'/(method+'_highlight.png'))
        ax.imshow(im);ax.contour(mask,levels=[.5],colors=['#00dcf5'],linewidths=.85)
        ax.set_title(f'({chr(97+i)}) {label}',loc='left',pad=6);ax.axis('off')
        # Magnify only real rendered pixels, with the same target crop in both panels.
        yy,xx=np.where(mask);pad=28
        x0=max(0,int(xx.min())-pad);x1=min(im.width,int(xx.max())+pad)
        y0=max(0,int(yy.min())-pad);y1=min(im.height,int(yy.max())+pad)
        zoom=ax.inset_axes([.62,.60,.36,.38]);zoom.imshow(im);zoom.contour(mask,levels=[.5],colors=['#00dcf5'],linewidths=1)
        zoom.set_xlim(x0,x1);zoom.set_ylim(y1,y0);zoom.set_xticks([]);zoom.set_yticks([])
        for s in zoom.spines.values():s.set_visible(True);s.set_linewidth(.5);s.set_color('#777777')
        zoom.set_title('Intended target',fontsize=6,pad=2)
        if i==0:
            inset=ax.inset_axes([.02,.02,.42,.45]);inset.imshow(Image.open(source/'native_text/text_only_highlight.png'))
            inset.set_title('Selected elsewhere',fontsize=6,pad=2);inset.set_xticks([]);inset.set_yticks([])
            for s in inset.spines.values():s.set_visible(True);s.set_linewidth(.5);s.set_color('#777777')
    fig.legend(handles=[Line2D([0],[0],color='#de4943',lw=4,label='Selected Gaussians'),
        Line2D([0],[0],color='#00c6e3',lw=1.4,label='Intended target')],loc='center',bbox_to_anchor=(.5,.515),ncol=2,frameon=False)
    left=fig.add_axes([.075,.105,.49,.30]);axes.append(left)
    n=len(rows);x=np.arange(1,n+1);colors=['#c58439','#226b99'];names=['Text-only','Photo-assisted']
    offsets=[-.17,.17] if supplied else [-.09,.09]
    for method,color,label,offset in zip(['text_only','photo_assisted'],colors,names,offsets):
        vals=np.array([r[method+'_'+suffix] if r[method+'_'+suffix] is not None else np.nan for r in rows])
        left.plot(x+offset,vals,'o',ms=3.3 if supplied else 4,color=color,label=label,clip_on=False)
    for i,row in enumerate(rows):
        a=row['text_only_'+suffix];b=row['photo_assisted_'+suffix]
        if a is not None and b is not None:left.plot([i+1+offsets[0],i+1+offsets[1]],[a,b],color='#aaa',lw=.7,zorder=0)
        else:left.text(i+1,.065,'NA',ha='center',fontsize=6,color='#777')
    left.set_title('(c) Localization IoU across 20 Cases' if supplied else ('(c) Held-out IoU (AI draft masks)' if draft else '(c) Held-out Localization IoU'),loc='left',pad=7)
    left.set(xlim=(.5,n+.5),ylim=(0,1.03),ylabel='IoU',xlabel='M1 case' if supplied else 'Confirmed M1 case')
    left.set_xticks(x,[f'{n:02d}'+('*' if draft and n==6 else '') for n in x]);left.set_yticks([0,.25,.5,.75,1]);left.grid(axis='y',alpha=.17)
    if supplied:
        left.tick_params(axis='x',labelsize=5.5,pad=4)
        left.legend(frameon=False,fontsize=6.3,loc='upper center',ncol=2,columnspacing=1,handletextpad=.4,borderpad=.15)
    elif not any(r[count_key] for r in rows):
        left.text(5.5,.61,'Independent masks pending',ha='center',fontsize=8,color='#555')
        left.text(5.5,.43,'0 / 20 reviewed test views',ha='center',fontsize=7,color='#555')
    else:
        left.legend(frameon=False,fontsize=6.7,loc='upper right')
        if draft:left.text(.98,.49,f'{summary["ai_draft_views"]} views / {summary["ai_draft_cases"]} cases',transform=left.transAxes,ha='right',fontsize=6.5,color='#555')
    right=fig.add_axes([.69,.105,.285,.30]);axes.append(right)
    for i,(method,color) in enumerate(zip(['text_only','photo_assisted'],colors)):
        s=summary[method];rate=s['case_selection_rate']
        if rate is not None:
            right.bar(i,rate,.52,color=color)
            right.text(i,rate+.035,f'{s["correct"]}/{s["valid_cases"]}',ha='center',fontsize=9,fontweight='bold')
    right.set_title('(d) Correct Instance Selection',loc='left',pad=7)
    right.set_xticks([0,1],['Text-only','Photo-assisted']);right.tick_params(axis='x',labelsize=6.8)
    right.set(ylim=(0,1.20),ylabel='Selection rate');right.set_yticks([0,.25,.5,.75,1]);right.yaxis.set_major_formatter(PercentFormatter(1));right.grid(axis='y',alpha=.17);right.set_axisbelow(True)
    right.text(.5,1.17,'AI visual audit',ha='center',va='top',fontsize=6.5,color='#555')
    fig.savefig(D/'figure_F08_revised.png',dpi=300,facecolor='white')
    fig.savefig(D/'figure_F08_revised.pdf',facecolor='white')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for i,ax in enumerate(axes):
        box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025,1.07)
        fig.savefig(panels/f'{chr(97+i)}.png',bbox_inches=box,dpi=300,facecolor='white')
    plt.close(fig)
    heldout_caption=('(c) Held-out localization IoU is currently unavailable: two non-input registered views per case (20 annotation tasks) have no independent human-reviewed masks; NA denotes missing data, not zero. ' if not summary['independent_human_masks'] else
        f"(c) Held-out localization IoU, averaged over independently reviewed test masks per target; {summary['independent_human_masks']} valid test views. Missing targets are marked NA. ")
    if draft:
        heldout_caption=(f'(c) Preliminary agreement with the supplied AI polygon drafts on {summary["ai_draft_views"]} held-out views from M1-03, M1-04, M1-05 and M1-06; each point is the mean over two views of one target. '
            'These masks are not human-verified ground truth. The remaining twelve empty masks are excluded and their six cases remain NA. '
            '* M1-06 includes a suspected instance mismatch in frame_000202; its supplied draft is retained unchanged for this provisional calculation. '
            f'Target-averaged draft IoU is {summary["text_only"]["ai_draft_mean_iou"]:.4f} for text-only and {summary["photo_assisted"]["ai_draft_mean_iou"]:.4f} for photo-assisted localization. ')
    if supplied:
        heldout_caption=(f'(c) Paired localization IoU for the 20 case labels in the author-supplied result table, including the corrected value for case 06. '
            f'Arithmetic means of the supplied case scores are {summary["text_only"]["reported_mean_iou"]:.4f} and {summary["photo_assisted"]["reported_mean_iou"]:.4f}, respectively. '
            'These scores are plotted as supplied rather than re-evaluated from per-view masks in this figure-production step. '
            'Panel (c) uses 20 reported cases; panel (d) retains the earlier ten-case instance-selection audit and its original denominator. ')
    caption=('Fig. 8. Reference-image-assisted instance localization in an industrial scene. '
        '(a-b) M1-10 (the previously displayed Case10): actual selected Gaussians are tinted red and the user-designated input region is outlined in cyan, '
        'using the same registered main camera. Insets show the intended target and, in (a), the text-selected object in its own registered inspection view. '
        +heldout_caption+'Input-SAM agreement is retained only as an auxiliary diagnostic. '
        '(d) Correct-instance selection in the ten confirmed M1 tasks is 4/10 for text-only and 10/10 for photo-assisted localization, based on an AI visual audit of photographs, multi-view Gaussian projections and existing user-confirmed target identities; independent human adjudication remains pending. '
        'M1-04 and M1-09 refer to the same physical cooler, so the ten tasks contain nine unique physical targets. Grouped results are 3/9 and 9/9, respectively. '
        'Both methods use identical per-case text and frozen three-scale LangSplat fields; text-only uses the existing geometry-based connected-component selector without reference-SAM or instance-position conditioning. '
        'The cohort consists of previously confirmed M1 cases in one scene. The saved descriptions for some cases, including M1-10, do not precisely match the selected equipment, so this comparison includes description mismatch as well as instance ambiguity. '
        'Test views are held out from instance-specific photo assistance, not from original scene/language-field training. No localization, geometry or opacity was altered for visualization.')
    if supplied:
        caption=('Fig. 8. Reference-image-assisted instance localization in an industrial scene. '
            '(a-b) Existing Case10 comparison using the same registered main camera: selected Gaussians are tinted red and the intended target is outlined in cyan. Insets show target details and the text-selected object elsewhere in the scene. '
            '(c) Localization IoU for 20 case labels from the author-supplied result table. Paired points are connected within each case; case 06 uses the updated author-provided value. '
            f'The arithmetic means are {summary["text_only"]["reported_mean_iou"]:.4f} for text-only and {summary["photo_assisted"]["reported_mean_iou"]:.4f} for photo-assisted localization. '
            'These supplied scores were not recomputed from per-view masks during this figure update. '
            '(d) The unchanged earlier AI visual audit of the original ten confirmed M1 tasks gives correct-instance selection counts of 4/10 and 10/10. '
            'Panels (c) and (d) therefore summarize different cohorts: twenty reported case scores versus the original ten-task instance audit. '
            'No Gaussian parameters, camera views, or qualitative images were changed.')
    (D/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    sources=read(D/'sources.json');sources.update(qualitative_case=record['m1_case_id'],qualitative_main_camera=record['reference_camera'],
        qualitative_inset_camera=record['text_native_camera'],main_view_crop='none; identical full reference image for both methods',
        target_inset_crop=[x0,y0,x1,y1],input_mask_is_only_visual_target_identifier=True,
        figure_size_mm=[180,129],raster_dpi=300,vector_pdf_text=True,statistics='metrics.json',review='instance_audit.json',
        ai_draft_iou=draft,user_supplied_iou=supplied,source_figure_directory=str(source_dir))
    save(D/'sources.json',sources)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(D/'figure_F08_revised.pdf'),str(D/'pdf_proof')],check=True)
    Image.open(D/'figure_F08_revised.png').convert('RGB').resize((1200,860),Image.Resampling.LANCZOS).save(D/'preview.jpg',quality=95)
    code=D/'code';code.mkdir(exist_ok=True)
    for name in ['evaluate_f08_ten.py','summarize_f08_ten.py','figure_f08_ten.py','f08_instance_audit.json']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    print('F08 figure exported: 180 x 129 mm, 300 DPI; evaluation mode: '+('user-supplied results' if supplied else ('AI draft mask agreement' if draft else 'independent annotation evaluation')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path);parser.add_argument('--source-dir',type=Path)
    args=parser.parse_args();main(args.output_dir,args.source_dir)
