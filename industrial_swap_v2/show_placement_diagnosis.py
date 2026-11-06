"""Plot measured diagnostic renders without altering image content."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostics',type=Path,default=Path(__file__).parent/'webapp/data/diagnostics')
    args=parser.parse_args(); root=args.diagnostics
    control=root/'m3_height_control_v1'; report=json.loads((control/'diagnosis.json').read_text())
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    entries=[
        ('独立高斯模型\n底座完整',control/'original_isolated.png'),
        ('移除旧设备后的背景\n本身存在拖影',root/'m3_placement/background.png'),
        ('原 M3 合成\n底部可见贡献 42.0%',control/'original_scene.png'),
        ('仅修正高度：诊断对照\n底部可见贡献 92.4%',control/'height_only_control_scene.png'),
    ]
    # Shared image coordinates; include both the original and lifted base.
    (xmin,ymin),(xmax,ymax)=report['base_projected_bbox']
    x0,x1=max(0,int(xmin-65)),min(1024,int(xmax+65))
    y0,y1=max(0,int(ymin-110)),min(1024,int(ymax+50))
    fig,axes=plt.subplots(2,4,figsize=(16,8),dpi=150,gridspec_kw={'height_ratios':[1,.78]})
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.1,wspace=.045,hspace=.04)
    for index,(title,path) in enumerate(entries):
        image=np.asarray(Image.open(path).convert('RGB'))
        axes[0,index].imshow(image)
        axes[0,index].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#19a9b6',linewidth=1))
        axes[0,index].set_title(title,fontproperties=font,fontsize=12,pad=10)
        axes[1,index].imshow(image[y0:y1,x0:x1])
        for row in range(2): axes[row,index].axis('off')
    fig.suptitle('M3 底座问题定位：同一相机、同一模型，分离背景与高度因素',fontproperties=font,fontsize=17,y=.97)
    fig.text(.02,.045,'下排为同一画面区域放大。右侧只沿已有地面法线平移，未重新拟合，也未替换原实验。',fontproperties=font,fontsize=11)
    fig.savefig(control/'comparison.png',facecolor='white')
    plt.close(fig)
    print(control/'comparison.png')


if __name__=='__main__': main()
