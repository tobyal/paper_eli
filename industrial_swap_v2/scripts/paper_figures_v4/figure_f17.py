"""Compose F17 from real common-camera projections; no model inference."""
import json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F17_localization_comparison'
METHODS=['GaussianEditor','DGE','Ours']
COLORS=['#be7d34','#697b8f','#25799a']
# Chosen for instance ambiguity and open-frame partial occlusion, respectively;
# selection is fixed independently of the measured method ordering.
QUAL=['M1-03_frame_000211','M1-05_frame_000228']

def crop_from_gt(mask):
    yy,xx=np.where(mask);h,w=mask.shape
    cw=max(320,(xx.max()-xx.min()+1)*2.6,(yy.max()-yy.min()+1)*2.1*4/3)
    cw=min(cw,w,h*4/3);ch=cw*3/4
    x0=min(max((xx.min()+xx.max())/2-cw/2,0),w-cw)
    y0=min(max((yy.min()+yy.max())/2-ch/2,0),h-ch)
    return tuple(round(v) for v in [x0,y0,x0+cw,y0+ch])

def main():
    d=json.loads((OUT/'metrics.json').read_text());summary={x['method']:x for x in d['summary']}
    valid=d['common_targets'];scores={(x['task_id'],x['method']):x for x in d['per_view']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.labelsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(180/25.4,133/25.4),facecolor='white');crops={};axes=[]
    x=[.013,.260,.507,.754];width=.233
    for i,label in enumerate(['Ground Truth*']+METHODS):fig.text(x[i]+width/2,.982,label,ha='center',va='top',fontsize=8.5,fontweight='bold')
    for row,tid in enumerate(QUAL):
        assert scores[(tid,'Ours')]['primary_included']
        gt=np.asarray(Image.open(OUT/'ground_truth_masks'/tid/'mask.png'))>127
        crop=crop_from_gt(gt);crops[tid]=dict(crop=list(crop),policy='GT-centered shared context crop; identical across all four columns',outside_crop_prediction_pixels={})
        y=.689-row*.308
        fig.text(.014,y+.244,f'({chr(97+row)}) '+tid.split('_')[0]+('  |  Similar coolers' if row==0 else '  |  Partly occluded assembly stand'),fontsize=7.2,va='bottom')
        for col,method in enumerate(['GT']+METHODS):
            im=Image.open(OUT/'panels'/f'{tid}_{method}.png')
            ax=fig.add_axes([x[col],y,width,.233]);axes.append(ax)
            ax.imshow(im.crop(crop));ax.axis('off')
            # Preserve visibility of false positives outside the shared target
            # crop; all four columns use the same unaltered full-photo inset.
            inset=ax.inset_axes([.008,.008,.36,.27])
            inset.imshow(im);inset.set_xticks([]);inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.7)
            im.crop(crop).save(OUT/'panels'/f'figure_{tid}_{method}.png',dpi=(300,300))
            if method=='GT':label='Independent AI draft'
            else:
                score=scores[(tid,method)]['primary_heldout_iou'];label=f'IoU = {score:.3f}'
                pred=np.asarray(Image.open(OUT/'projected_masks'/f'{tid}_{method}.png'))>127
                crops[tid]['outside_crop_prediction_pixels'][method]=int(pred.sum()-pred[crop[1]:crop[3],crop[0]:crop[2]].sum())
            fig.text(x[col]+width/2,y-.009,label,ha='center',va='top',fontsize=7)
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=4,label='Selected Gaussians'),
        Line2D([0],[0],color='#00d2e6',lw=1.4,label='GT contour (AI draft)')],loc='center',bbox_to_anchor=(.5,.329),ncol=2,frameon=False,fontsize=6.7)
    ax=fig.add_axes([.084,.074,.895,.20]);axes.append(ax)
    xpos=np.arange(len(valid));bw=.21
    for j,(method,color) in enumerate(zip(METHODS,COLORS)):
        vals=[next(r['heldout_iou'] for r in d['per_target'] if r['label']==label and r['method']==method) for label in valid]
        ax.bar(xpos+(j-1)*bw,vals,bw*.93,color=color,label=f"{method} (mean {summary[method]['mean_heldout_iou']:.3f})",zorder=3)
    ax.set_xticks(xpos,valid);ax.set_ylim(0,1.12);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Held-out IoU');ax.set_xlabel('Physical equipment target')
    ax.grid(axis='y',color='#e9e9e9',zorder=0)
    ax.set_title('(c) Localization IoU on common eligible targets',loc='left',pad=6)
    ax.legend(loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.05),columnspacing=1.4,handlelength=1.3)
    fig.savefig(OUT/'figure_F17.png',dpi=300);fig.savefig(OUT/'figure_F17.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.01,1.04)
    fig.savefig(OUT/'panels/c_statistics.png',bbox_inches=box,dpi=300);plt.close(fig)
    first=d['summary'][0];nviews=first['common_valid_views'];ntargets=first['common_valid_targets']
    means=', '.join(f"{m}: {summary[m]['mean_heldout_iou']:.4f}" for m in METHODS)
    views=json.loads((OUT/'evaluation_views.json').read_text())['views']
    excluded=[v for v in views if v['ai_draft_nonempty'] and not v['primary_included']]
    exclusion='; '.join(v['task_id']+' ('+', '.join(v['exclusion_reasons'])+')' for v in excluded)
    caption=('Fig. 17. Quantitative comparison of industrial equipment localization. '
        '(a-b) Actual selected Gaussians from GaussianEditor, DGE and reference-image-guided LangSplat (Ours) are projected into the same registered test photograph for each row. '
        'Red indicates visible selected-Gaussian contribution; cyan indicates the independent AI polygon draft (*Ground Truth column denotes provisional annotation, not human-verified ground truth). '
        'The two rows illustrate similar cooler instances and a partly occluded assembly stand. Each row shares one camera and crop. The lower-left inset in every image preserves the complete photograph and all projected selections, including false positives outside the enlarged target region. IoU is computed over the full uncropped 938 x 528 image, so off-crop false positives remain penalized. '
        '(c) Per-target IoU averages the eligible views of each target. '
        f'The preliminary common subset contains {nviews} views of {ntargets} distinct physical targets in one scene. Target-averaged means are {means}. '
        'Five of the nine physical targets have no eligible independent annotations and are not scored as zero. '
        f'The eight supplied nonempty drafts are all retained and scored for sensitivity analysis; primary exclusions are {exclusion}. '
        'All primary test cameras are absent from both baseline tracing samples and the corresponding photo-assistance input; they are not held out from original scene or language-field training. '
        'GE and DGE receive the same target Text A; Ours additionally receives a registered photo specifying the intended instance. M1-03 and M1-06 have identical text and therefore share each baseline\'s text-only prediction. '
        'All methods use the same original scene, Gaussian rasterizer, full-scene occlusion and visible-contribution threshold 0.05. No generation, editing, language-field retraining or simulated result table is used. '
        'AI draft annotations and instance identities require independent human adjudication before definitive accuracy claims.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    sources=json.loads((OUT/'sources.json').read_text());sources.update(figure_mm=[180,133],dpi=300,vector_pdf_text=True,
        qualitative_selection='Fixed before comparing measured scores: similar coolers M1-03/frame211 and partly occluded frame M1-05/frame228',qualitative_crops=crops,
        qualitative_insets='Identical lower-left full-photo overview in all four columns, with corresponding real selections and GT contour')
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17.pdf'),str(OUT/'pdf_proof')],check=True)
    print('F17 figure:',ntargets,'targets /',nviews,'views;',means,flush=True)

if __name__=='__main__':main()
