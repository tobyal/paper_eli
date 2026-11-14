"""Square F17 panels, vector GT outlines and a moderately upright mean chart.

Presentation only. Binary GT, evaluation data and actual Gaussian contributions
are preserved; the six user-polygon vertices are drawn as antialiased vectors.
"""
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
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
OUT=BASE/'F17_localization_comparison_final'
BACKUP=BASE/'F17_localization_comparison_final_backup_20261111_square_chart'
NATIVE=BASE/'F17_localization_comparison'
TASK='M1-03_frame_000211'
METHODS=['GaussianEditor','DGE','Ours']
MAIN_CROP=[633,197,753,317]
ONLY_CROP=[630,194,762,326]
SCALE=3

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def picture(a):return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))


def main():
    if not BACKUP.exists():
        assert (OUT/'inputs/GT_extraction.json').is_file()
        assert OUT.resolve().parent==BASE.resolve()==BACKUP.resolve().parent
        OUT.rename(BACKUP)
    for folder in ['panels','renders','code','ground_truth_masks','inputs']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    original_hashes={str(p.relative_to(BACKUP)):sha(p) for p in BACKUP.rglob('*') if p.is_file()}
    for name in ['iou_summary.csv','metrics.json','metric_changes.json','evaluation_views.json']:
        shutil.copy2(BACKUP/name,OUT/name)
    for folder in ['inputs','ground_truth_masks']:
        shutil.copytree(BACKUP/folder,OUT/folder,dirs_exist_ok=True)
    data=read(OUT/'metrics.json')
    gt_source=read(OUT/'inputs/GT_extraction.json')
    previous=read(BACKUP/'sources.json')
    scores={r['method']:r['primary_heldout_iou'] for r in data['per_view'] if r['task_id']==TASK}
    means={r['method']:r['mean_heldout_iou'] for r in data['summary']}
    vertices=np.array(gt_source['polygon_vertices_evaluation'])
    photo=Image.open(NATIVE/'photos'/f'{TASK}.png').convert('RGB')
    raw=np.asarray(photo.resize((photo.width*SCALE,photo.height*SCALE),Image.Resampling.LANCZOS))/255.
    rgb={'GT':raw}
    raster_sources=[]
    for method in METHODS:
        cp=NATIVE/'projected_masks_revised'/f'{TASK}_{method}_display_contribution_16bit.png'
        alpha=.8*np.asarray(Image.open(cp),dtype=np.float32)/65535.
        rgb[method]=raw*(1-alpha[...,None])+np.array([1.,.10,.075])*alpha[...,None]
        full=BACKUP/'renders'/f'{method}_gaussian_only_full.png'
        shutil.copy2(full,OUT/'renders'/full.name)
        Image.open(full).crop(tuple(v*SCALE for v in ONLY_CROP)).save(OUT/'panels'/f'{method}_gaussian_only.png',dpi=(300,300))
        raster_sources.append(dict(method=method,contribution=str(cp),contribution_sha256=sha(cp),
            selected_only_render=str(full),selected_only_sha256=sha(full)))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    scene_images={}
    polygon_points=(vertices-np.array(MAIN_CROP[:2]))*SCALE
    def scene(ax,method):
        ax.imshow(scene_images[method],interpolation='lanczos')
        ax.add_patch(Polygon(polygon_points,closed=True,fill=False,edgecolor='#00D2E6',
            linewidth=.85,antialiased=True,joinstyle='round',capstyle='round',rasterized=False))
        ax.set_xlim(-.5,MAIN_CROP[2]*SCALE-MAIN_CROP[0]*SCALE-.5)
        ax.set_ylim(MAIN_CROP[3]*SCALE-MAIN_CROP[1]*SCALE-.5,-.5)
        ax.axis('off')
    for method in ['GT']+METHODS:
        full=picture(rgb[method]);full.save(OUT/'renders'/f'{method}_scene_without_contour.png')
        scene_images[method]=full.crop(tuple(v*SCALE for v in MAIN_CROP))
        # Standalone square panel: same vector path, downsampled by the renderer
        # rather than tracing the stair-stepped binary GT pixels.
        pf=plt.figure(figsize=(31.5/25.4,31.5/25.4),facecolor='white')
        scene(pf.add_axes([0,0,1,1]),method)
        pf.savefig(OUT/'panels'/f'{method}_scene.png',dpi=600)
        plt.close(pf)
    # Extend only the right-hand canvas; all left-hand image sizes/positions
    # remain unchanged in millimetres and in the 300-DPI export.
    w,h=190.,88.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    text(1.5,85.5,'(a) M1-03 | Similar coolers',ha='left',va='center',fontsize=8.5)
    cols=[1.5,35.,68.5,102.];pw=31.5
    for i,method in enumerate(['GT']+METHODS):
        text(cols[i]+pw/2,80.8,'2D GT' if method=='GT' else method,ha='center',va='center',fontsize=7.8,fontweight='bold')
        scene(axis(cols[i],46,pw,pw),method)
        text(cols[i]+pw/2,43.4,'Updated GT polygon' if method=='GT' else f'IoU = {scores[method]:.3f}',ha='center',va='center',fontsize=7.1)
        if method!='GT':
            text(cols[i]+pw/2,39.2,'Gaussian-only',ha='center',va='center',fontsize=6.8)
            ax=axis(cols[i],4.5,pw,pw)
            ax.imshow(Image.open(OUT/'panels'/f'{method}_gaussian_only.png'));ax.axis('off')
    ax=axis(cols[0],4.5,pw,pw);ax.axis('off')
    ax.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.5,label='Selected Gaussians\n(projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.3,label='GT contour')],loc='center',frameon=False,
        fontsize=6,handlelength=1.3,labelspacing=1.3,handletextpad=.6,borderpad=0)
    # Enlarge both dimensions without making the chart excessively tall.
    chart_bounds=[134.,14.,190.,80.]
    ax=axis(145,21,43.5,46)
    values=[means[m] for m in METHODS]
    ax.bar(range(3),values,.55,color=['#be7d34','#697b8f','#25799a'],zorder=3)
    for i,v in enumerate(values):ax.text(i,v+.028,f'{v:.3f}',ha='center',va='bottom',fontsize=8,fontweight='bold')
    ax.set_ylim(0,1);ax.set_xlim(-.55,2.55);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylabel('IoU',fontsize=7.3,labelpad=1.5)
    ax.set_xticks(range(3),['Gaussian\nEditor','DGE','Ours']);ax.tick_params(axis='both',labelsize=6.9,length=2,pad=2)
    ax.grid(axis='y',color='#e9e9e9',zorder=0)
    text(167,75.4,'(b) Mean Held-out\nLocalization IoU',ha='center',va='center',fontsize=8.2,linespacing=1.1)
    text(167,70,'4 devices / 6 test views',ha='center',va='center',fontsize=6.5,color='#444444')
    assert len(ax.patches)==3 and len(ax.collections)==0
    fig.savefig(OUT/'figure_F17_final.png',dpi=300)
    fig.savefig(OUT/'figure_F17_final.pdf')
    fig.savefig(OUT/'panels/mean_iou.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in chart_bounds)),dpi=600)
    plt.close(fig)
    caption=(BACKUP/'caption.txt').read_text(encoding='utf-8')
    caption=caption.replace('in a 4:3 chart block','in a moderately upright chart')
    caption=caption.replace('The M1-03 similar-cooler example','(a) The M1-03 similar-cooler example').replace('Right: mean held-out','(b) Mean held-out')
    caption+='\nPanel (b) is enlarged in both dimensions; panel (a), the GT, and all experimental values remain unchanged.\n'
    (OUT/'caption.txt').write_text(caption,encoding='utf-8')
    write(OUT/'sources.json',dict(revision='Enlarged mean chart labelled (b); square qualitative panel (a) unchanged',
        previous_final_backup=str(BACKUP),previous_sources=previous,updated_GT=gt_source,
        raster_sources=raster_sources,main_crop=MAIN_CROP,gaussian_only_crop=ONLY_CROP,
        main_panel_mm=[pw,pw],gaussian_only_panel_mm=[pw,pw],figure_mm=[w,h],dpi=300,
        GT_display='Saved continuous polygon vertices; no binary-edge tracing. Vector PDF, antialiased PNG, round joins.',
        GT_contour_linewidth_points=.85,GT_evaluation_mask_unchanged=True,
        scene_red_RGB=[1.,.10,.075],gaussian_only_red_RGB=[1.,.19,.10],
        chart_mm=[56.,66.],chart_plot_mm=[43.5,46.],chart_label='(b)',
        previous_chart_plot_mm=[33.5,38.],left_panel_size_and_positions_unchanged=True,
        mean_IoU=means,qualitative_IoU=scores,
        camera_and_predictions_unchanged=True,valid_targets=4,valid_views=6))
    assert all(sha(BACKUP/p)==v for p,v in original_hashes.items())
    for name in ['iou_summary.csv','metrics.json','evaluation_views.json','ground_truth_masks/'+TASK+'.png']:
        assert sha(OUT/name)==sha(BACKUP/name)
    dimensions={p.name:list(Image.open(p).size) for p in (OUT/'panels').glob('*.png')}
    assert all(v[0]==v[1] for k,v in dimensions.items() if k!='mean_iou.png')
    write(OUT/'validation.json',dict(backup_unchanged=True,GT_and_metrics_byte_identical=True,
        all_seven_image_panels_square=True,vector_GT_contour=True,mean_bar_count=3,scatter_count=0,
        no_models_or_training=True,source_rendering_reused=True,panel_pixels=dimensions,
        figure_pixels=list(Image.open(OUT/'figure_F17_final.png').size)))
    write(OUT/'backup_manifest.json',dict(directory=str(BACKUP),sha256=original_hashes))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F17 - square panels</title>
<style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F17 · 放大的 (b) 统计图</h1><p>右侧统计图加大并标注 (b)；左侧方形图、平滑 GT 轮廓、IoU、高斯与相机均保持不变。</p>
<p><a href="figure_F17_final.png">300 DPI PNG</a> · <a href="figure_F17_final.pdf">PDF</a> · <a href="iou_summary.csv">IoU CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F17_localization_comparison_final_backup_20261111_square_chart/index.html">上一版</a></p>
<img src="figure_F17_final.png">'''
    (OUT/'index.html').write_text(page+'<p>'+html.escape(caption)+'</p>',encoding='utf-8')
    print('Square F17 complete: all seven panels square; vector GT; original GT/metrics/backup preserved.',flush=True)

if __name__=='__main__':main()
