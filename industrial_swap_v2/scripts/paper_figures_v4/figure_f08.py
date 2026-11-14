import numpy as np
from figure_common import *

def main():
    d=folder('F08_localization');raw=read(d/'localization_metrics.json');rows=[r for r in raw if r['method'] in ('gsle','ours')]
    fig=plt.figure(figsize=(9,6.7));gs=fig.add_gridspec(2,2,height_ratios=[1,1.25],left=.095,right=.97,bottom=.16,top=.88,wspace=.35,hspace=.5)
    fig.suptitle('Reference-Image-Assisted Instance Localization',y=.97,fontsize=14,fontweight='bold')
    fig.text(.5,.915,'Input-SAM agreement diagnostic; independent human annotations unavailable',ha='center',fontsize=9,color='#555555')
    axes=[];a=fig.add_subplot(gs[0,0]);axes.append(a)
    for i,case in enumerate([r['case'] for r in CASES]):
        vals=[next(r['reference_sam_iou'] for r in rows if r['case']==case and r['method']==m) for m in ('gsle','ours')]
        a.plot([0,1],vals,'o-',color=COLORS[i],label=case,lw=1.5)
    a.set_xticks([0,1],['Text-only','Photo-assisted']);a.set(ylabel='Visible-mask / reference SAM IoU',ylim=(-.02,1.02),title='(a) Reference-region overlap');a.legend(frameon=False,fontsize=7);a.grid(axis='y',alpha=.2)
    b=fig.add_subplot(gs[0,1]);axes.append(b)
    for i,m in enumerate(['gsle','ours']):
        vals=[r for r in rows if r['method']==m]
        means=[np.mean([r[k] for r in vals]) for k in ['reference_sam_recall','reference_sam_precision']]
        b.bar(np.arange(2)+(i-.5)*.3,means,.3,color=COLORS[i],label=['Text-only','Photo-assisted'][i])
        for k,name in enumerate(['reference_sam_recall','reference_sam_precision']):
            b.scatter(np.full(len(vals),k+(i-.5)*.3),[r[name] for r in vals],s=9,c='black',alpha=.5)
    b.set_xticks([0,1],['Target-region recall','Target-region precision']);b.set(ylim=(0,1.05),title='(b) In-sample region statistics');b.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2);b.grid(axis='y',alpha=.2)
    case='case10'
    for i,(m,label) in enumerate([('gsle','Text-only'),('ours','Photo-assisted')]):
        a=fig.add_subplot(gs[1,i]);axes.append(a);im=np.array(Image.open(d/'cases'/case/(m+'_highlight.png')))
        a.imshow(im);ref=np.array(Image.open(d/'cases'/case/'reference_mask.png'))>0;a.contour(ref,levels=[.5],colors=['#00DDFF'],linewidths=.7)
        a.set_title('(c) '+case+' - '+label);a.axis('off')
        if m=='gsle':
            native=ROOT/f'baseline_comparison/gsle_native/{case}/localization/view_0_selected.png'
            inset=a.inset_axes([.02,.57,.38,.40]);inset.imshow(Image.open(native));inset.set_title('Selected elsewhere: native view',fontsize=6);inset.axis('off')
    fig.text(.095,.055,'Red: actual selected Gaussians; cyan: input reference SAM. All selections retain their native geometry and opacity.\nText-only uses the existing three-scale LangSplat component selector. SAM is an input to photo assistance;\nthese measurements quantify reference adherence and do not establish held-out localization accuracy.',fontsize=8)
    caption='Figure 8. Reference-image-assisted instance localization across five existing industrial cases. Text-only reuses the actual GSLE-native LangSplat query and geometry-only connected-component ranking without reference-image/SAM conditioning. Photo-assisted reuses the confirmed M1 selections. All selected Gaussians, including disconnected errors, are rendered with their original opacity in the same registered reference camera per case; visible membership uses contribution > 0.05. (a) Paired input-SAM overlap. (b) Means and individual-case values for reference-region recall and precision. (c) Case10 illustrates the largest observed instance-selection discrepancy, with semi-transparent red selections and cyan reference contours. These SAM regions were used by the photo-assisted method; they are not independent ground truth. Independent instance-selection accuracy and held-out annotated IoU remain unevaluated.'
    caption+=' The inset shows the text-only selected instance in its saved native inspection camera; the main comparison panels and metrics use the same registered reference camera.'
    provenance=read(d/'evaluation_sources.json');provenance['native_instance_inset']=source(native)
    export(fig,d,'F08',caption,provenance,rows,axes)

if __name__=='__main__':main()
