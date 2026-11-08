"""Show real saved renders and pixel differences; never modifies scene renders."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preview',type=Path,required=True); args=p.parse_args()
    folder=args.preview; report=json.loads((folder/'occlusion_report.json').read_text())
    before=np.asarray(Image.open(folder/'validation/00_avp_099_before.png'))
    after=np.asarray(Image.open(folder/'validation/00_avp_099_after.png'))
    mask=np.asarray(Image.open(folder/'renders/00_object_mask.png').resize((before.shape[1],before.shape[0]),Image.Resampling.NEAREST))>127
    yy,xx=np.nonzero(mask); x0,x1=xx.min(),xx.max(); y0,y1=yy.min(),yy.max()
    # Generic lower half of the rendered target, only for close inspection.
    crop=(max(x0-35,0),int((y0+y1)/2),min(x1+35,before.shape[1]),min(y1+35,before.shape[0]))
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists(): plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    fig,axes=plt.subplots(2,2,figsize=(12,10),gridspec_kw={'height_ratios':[1,.58]})
    for column,(image,title) in enumerate(((before,'原 M3'),(after,'OAGOF：仅过滤背景遮挡'))):
        axes[0,column].imshow(image); axes[0,column].set_title(title,fontsize=15)
        axes[0,column].add_patch(Rectangle(crop[:2],crop[2]-crop[0],crop[3]-crop[1],fill=False,edgecolor='#21bdd0',linewidth=1.5))
        axes[1,column].imshow(image[crop[1]:crop[3],crop[0]:crop[2]])
        axes[1,column].set_title('同一区域放大',fontsize=12)
    for ax in axes.flat: ax.axis('off')
    fig.suptitle('目标感知遮挡过滤：同相机、同设备模型，不补地面',fontsize=17,y=.98)
    fig.text(.5,.025,f"删除 {report['removed_count']} 个背景高斯；新增 0 个高斯；设备和其余背景参数逐字节不变。\n"
             '证据不足的遮挡保留，设备 Mask 外的黑色残影不作整体修复。',ha='center',fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.09,wspace=.04,hspace=.1)
    fig.savefig(folder/'same_view_comparison.png',dpi=150,facecolor='white'); plt.close(fig)
    # This diagnostic was computed with the exact full-resolution mask by
    # the independent renderer audit; do not re-estimate its mask in 2D.
    difference=np.asarray(Image.open(folder/'validation/00_avp_099_outside_change.png').convert('L'))/255
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    axes[0].imshow(mask,cmap='gray'); axes[0].set_title('M3 独立渲染的设备 Mask')
    im=axes[1].imshow(difference,cmap='magma',vmin=0,vmax=.1)
    axes[1].set_title('Mask 外变化（绝对 RGB 差，0–0.1）')
    for ax in axes: ax.axis('off')
    fig.colorbar(im,ax=axes[1],fraction=.045); fig.tight_layout(); fig.subplots_adjust(top=.9)
    fig.savefig(folder/'outside_mask_difference.png',dpi=140); plt.close(fig)


if __name__=='__main__': main()
