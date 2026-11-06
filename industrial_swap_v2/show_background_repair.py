"""Arrange actual saved renders into a labeled scientific comparison figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preview-dir',type=Path,required=True)
    args=p.parse_args(); directory=args.preview_dir
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    report=json.loads((directory/'repair_report.json').read_text(encoding='utf-8'))
    before=Image.open(directory/'renders/avp_099_before.png')
    after=Image.open(directory/'renders/avp_099_after.png')
    crop=(325,600,705,850)
    fig,axes=plt.subplots(2,2,figsize=(13,11),gridspec_kw={'height_ratios':[1,.60]})
    for j,(im,title) in enumerate(((before,'修复前：原 M3 合成场景'),(after,'修复后：清理异常高斯 + 三维地面补全'))):
        axes[0,j].imshow(im); axes[0,j].set_title(title,fontsize=15,pad=12)
        axes[0,j].add_patch(Rectangle(crop[:2],crop[2]-crop[0],crop[3]-crop[1],fill=False,edgecolor='#14b8c8',linewidth=1.6))
        axes[1,j].imshow(im.crop(crop)); axes[1,j].set_title('同一区域放大',fontsize=12,pad=8)
    for ax in axes.flat: ax.axis('off')
    fig.suptitle('同一相机、同一机械臂、同一位置：仅修复三维背景',fontsize=18,y=.99)
    b=report['base_visibility_ratio']
    fig.text(.5,.025,f"移除 {report['removed_count']:,} 个异常背景高斯；机械臂参数逐字节不变。底部高斯可见贡献：{b['before']:.1%} → {b['after']:.1%}。\n"
        '画面为修复后 PLY 的真实渲染；地面补全来自周围地面，原始遮挡区域纹理无法精确恢复。',ha='center',va='bottom',fontsize=11)
    fig.subplots_adjust(left=.015,right=.985,top=.91,bottom=.095,wspace=.035,hspace=.08)
    fig.savefig(directory/'same_view_comparison.png',dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__': main()
