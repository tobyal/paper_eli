import numpy as np
from figure_common import *

def main():
    d=folder('F11_m2_ablation');rows=read(d/'metrics.json');fig=plt.figure(figsize=(10,8.7))
    gs=fig.add_gridspec(3,12,left=.055,right=.98,bottom=.16,top=.90,hspace=.42,wspace=.75,height_ratios=[1,1,.95]);axes=[]
    fig.suptitle('Ablation of Scene-Conditioned Image Generation',y=.97,fontsize=14,fontweight='bold')
    names=['C1: RGB + Text','C2: + Position Hint','C3: + Original Depth','C4: + Background Depth']
    mask=np.array(Image.open(d/'conditions/C1/m2/edit_mask.png'))>127;yy,xx=np.nonzero(mask)
    crop=[max(0,xx.min()-22),max(0,yy.min()-22),min(mask.shape[1],xx.max()+22),min(mask.shape[0],yy.max()+22)]
    for i,key in enumerate(['C1','C2','C3','C4']):
        image=Image.open(d/'conditions'/key/'m2/edited_scene.png');image.save(d/'panels'/(key+'_generated.png'));image.crop(crop).save(d/'panels'/(key+'_detail.png'))
        a=fig.add_subplot(gs[0,i*3:i*3+3]);axes.append(a);a.imshow(image);a.axis('off');a.set_title(names[i],fontsize=9)
        a=fig.add_subplot(gs[1,i*3:i*3+3]);axes.append(a);a.imshow(image.crop(crop));a.axis('off')
    for i,(metric,title,ylabel) in enumerate([('object_inside_edit_fraction','(a) Location adherence','Object area inside edit mask'),('background_rgb_mae','(b) Outside-edit RGB change','Mean absolute RGB difference')]):
        a=fig.add_axes([[.065,.17,.23,.19],[.40,.17,.21,.19]][i]);axes.append(a);a.bar(range(4),[r[metric] for r in rows],color=COLORS[:4]);a.set_xticks(range(4),['C1','C2','C3','C4']);a.set(title=title,ylabel=ylabel);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
        if i==0:a.set_ylim(0,1.05)
    a=fig.add_axes([.78,.17,.20,.19]);axes.append(a)
    values=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[np.nan]*4])
    a.imshow(np.ma.masked_invalid(values),cmap=plt.cm.Greens,vmin=0,vmax=1,aspect='auto');a.set_facecolor('#eeeeee')
    a.set_xticks(range(4),['C1','C2','C3','C4']);a.set_yticks(range(4),['Yellow body','Articulated arm','Square pedestal','Six actuated axes']);a.tick_params(axis='y',labelsize=7)
    for y in range(4):
        for x in range(4):a.text(x,y,'yes' if y<3 else 'N/E',ha='center',va='center',color='white' if y<3 else '#555555',fontsize=8)
    a.set_title('(c) Visual checklist',fontsize=9)
    for r in rows:r.update(visual_yellow_body=True,visual_articulated_arm=True,visual_square_pedestal=True,six_actuated_axes='not_verifiable_from_single_view',attribute_evaluator='Assistant visual inspection; descriptive, not independent accuracy')
    fig.text(.055,.06,'One generated image per condition, identical seed and target text; all four outputs are broadly similar.\nLocation is estimated with the same ROI-prompted SAM/CLIP procedure, not manual ground truth.\nChecklist: visible appearance only. Mechanical axis count/functionality is not verified (N/E).',fontsize=8)
    caption='Figure 11. Single-case qualitative conditioning ablation with Qwen-Image-2.1-Pro. All four new images use seed 42, the same original RGB/target description/camera, and the current adaptive editing mask for evaluation. Only the supplied position/depth conditions and the sentences describing those inputs differ. Top rows show the actual returned full scenes and identical pixel crops. (a) Fraction of estimated object-mask pixels inside the designated edit mask. SAM candidates use a shared ROI box and are ranked by SAM confidence times CLIP relevance, without containment filtering; this is not an independent object-location annotation. (b) RGB MAE relative to the original anchor outside the same edit mask. (c) A descriptive visual audit finds a yellow articulated arm and square pedestal in all four outputs. Six actuated axes and mechanical functionality cannot be established from these images. One sample per condition does not support statistical superiority claims; the observed outputs are similar.'
    export(fig,d,'F11',caption,dict(condition_policy=read(d/'condition_policy.json'),metrics=read(d/'metric_sources.json'),common_crop=[int(x) for x in crop],attribute_audit='Assistant visual inspection of all four actual images; no independent human labels'),rows,axes)

if __name__=='__main__':main()
