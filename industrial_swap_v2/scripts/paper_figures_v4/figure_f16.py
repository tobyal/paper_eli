import numpy as np
from figure_common import *

def box(mask,pad=25):
    y,x=np.nonzero(mask);return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(mask.shape[1],int(x.max())+pad),min(mask.shape[0],int(y.max())+pad)]

def main():
    d=folder('F16_failure_cases');fig,ax=plt.subplots(3,3,figsize=(9,9));fig.subplots_adjust(left=.04,right=.98,top=.91,bottom=.13,hspace=.32,wspace=.12)
    fig.suptitle('Observed Failure Modes and Applicability Limits',y=.97,fontsize=14,fontweight='bold');rows=[]
    loc=OUT/'F08_localization/cases/case02';reference=np.array(Image.open(loc/'reference_mask.png'))>0;selected=np.array(Image.open(loc/'ours_mask.png'))>0;b=box(reference,35)
    discrepancy=np.zeros((*reference.shape,3),dtype=np.uint8);discrepancy[selected&reference]=[235,235,235];discrepancy[selected&~reference]=[172,86,181];discrepancy[reference&~selected]=[235,140,30]
    loc_images=[Image.open(loc/'ours_highlight.png'),Image.open(loc/'ours_highlight.png').crop(b),Image.fromarray(discrepancy).crop(b)]
    gen=OUT/'F09_text_control/variants/C_cnc/m2';mask=np.array(Image.open(gen/'object_mask.png'))>0;gb=box(mask,15)
    gen_images=[Image.open(gen/'edited_scene.png').crop(gb),Image.open(gen/'object_white.png').crop(gb),Image.open(gen/'object_mask.png').convert('RGB').crop(gb)]
    integ=OUT/'F14_background_refinement/panels';old=np.array(Image.open(integ/'old_mask.png').resize((1024,1024),Image.Resampling.NEAREST))>0;new=np.array(Image.open(integ/'new_mask.png').resize((1024,1024),Image.Resampling.NEAREST))>0
    ib=box(old|new,35);ib[1]=int(ib[1]+(ib[3]-ib[1])*.48)
    supervision=np.zeros((1024,1024,3),np.uint8);supervision[old&~new]=[235,140,30];supervision[new]=[50,115,205]
    integ_images=[Image.open(integ/'full_candidate.png'),Image.open(integ/'full_candidate.png').crop(ib),Image.fromarray(supervision).crop(ib)]
    labels=[['(a) M1: Case02 selection','Bounding-box over-selection','Reference-mask discrepancy'],
            ['(b) M2: Generated CNC reference','Extracted foreground: display lost','Actual SAM mask'],
            ['(c) M4: Case00 candidate','Residual background / contact area','Old-only zone and new-object mask']]
    for i,images in enumerate([loc_images,gen_images,integ_images]):
        for j,image in enumerate(images):
            image.save(d/'panels'/f'failure_{i+1}_{j+1}.png');ax[i,j].imshow(image);ax[i,j].axis('off');ax[i,j].set_title(labels[i][j],fontsize=9)
    rows=[dict(failure='localization',case='case02',observation='Box-selected Gaussians extend beyond the input SAM region; purple is selection outside reference, orange is missed reference. SAM is not independent ground truth.',source=str(loc)),
          dict(failure='generation_condition',case='F09_C_cnc',observation='The Qwen image has a control display, but automatic SAM extraction removes much of the display face. This incomplete foreground is the actual TRELLIS input; no output was manually corrected.',source=str(gen)),
          dict(failure='integration',case='case00',observation='The 2000-step candidate improves base visibility but retains irregular old-device-region background. Old-only pixels are excluded from original-photo RGB supervision; hidden replacement background has no direct observation. Candidate fails preservation guard.',source=str(integ))]
    fig.text(.04,.055,'(a) White: shared mask; purple: selected outside input SAM; orange: reference region missed.\n(b) A real foreground-extraction failure upstream of 3D generation, not a fabricated mechanical defect.\n(c) Orange: old-only exclusion; blue: new-object mask. Existing scene parameters are unchanged.',fontsize=8)
    caption='Figure 16. Real limitations observed in the existing pipeline and the requested F09 run. (a) Case02 bounding-box localization includes Gaussians outside the selected SAM reference; the discrepancy is diagnostic because the SAM region is an input, not independent ground truth. (b) In the actual CNC generation run, the side control display is visible in the Qwen image but partly removed by automatic SAM foreground extraction, giving TRELLIS an incomplete condition. The extraction result is retained without manual repair or a replacement image. This is an upstream generation-condition failure, not evidence of a defective real machine. (c) The existing Case00 2,000-step M4 candidate retains irregular background near the old footprint despite improved base visibility. The old-only region lacks original-photo RGB supervision and unseen background cannot be reconstructed by color/opacity refinement alone. The candidate also fails the existing preservation guard. No failure was deliberately generated or any Gaussian altered for illustration.'
    sources=dict(localization=source(loc/'ours_highlight.png'),reference=source(loc/'reference_mask.png'),generation=source(gen/'edited_scene.png'),foreground=source(gen/'object_rgba.png'),mask=source(gen/'object_mask.png'),integration=read(OUT/'F14_background_refinement/render_sources.json'),annotation_crops={'localization':b,'generation':gb,'integration':ib})
    export(fig,d,'F16',caption,sources,rows,ax.ravel())

if __name__=='__main__':main()
