"""Publication layout of the actual Case02 A/B/C registration experiment."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration'
COLORS=['#7B8490','#D48938','#217D99']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
    'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':300})


def metric(ax,rows,key,title,percent=False):
    values=np.array([float(r[key]) for r in rows])*(100 if percent else 1)
    ax.bar(range(3),values,color=COLORS,width=.58,zorder=3)
    for i,value in enumerate(values):
        ax.annotate(f'{value:.2f}%' if percent else f'{value:.3f}',(i,value),
                    xytext=(0,5),textcoords='offset points',ha='center',fontsize=11)
    ax.set_xticks(range(3),['A','B','C'])
    ax.set_ylim(0,1.10 if key=='silhouette_iou' else max(float(values.max())*1.28,.01))
    ax.set_title(title,pad=12,loc='left',fontweight='bold')
    ax.yaxis.grid(True,color='#E6E8EB',linewidth=.7,zorder=0)
    ax.set_axisbelow(True)


def footprint(ax,points,title=True):
    low,high=points['low'],points['high']
    ax.add_patch(Rectangle((low[0],low[2]),high[0]-low[0],high[2]-low[2],
        fill=False,edgecolor='#282B30',linewidth=1.8,linestyle='--',label='Allowed workspace'))
    for k,color in zip('ABC',COLORS):
        xyz=points[k][:,[0,2]]
        hull=ConvexHull(xyz)
        ax.add_patch(Polygon(xyz[hull.vertices],closed=True,facecolor=color,
                            edgecolor=color,alpha=.13,linewidth=0))
        ax.add_patch(Polygon(xyz[hull.vertices],closed=True,fill=False,edgecolor=color,
                            linewidth=1.6,label=k))
    allp=np.concatenate([points[k][:,[0,2]] for k in 'ABC']+[np.array([[low[0],low[2]],[high[0],high[2]]])])
    lo=allp.min(0);hi=allp.max(0);margin=(hi-lo)*.12
    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Local x');ax.set_ylabel('Local z',labelpad=0)
    if title:ax.set_title('(g) Workspace Footprint',loc='left',pad=12,fontweight='bold')
    ax.grid(color='#E9EAEC',linewidth=.6)


def main():
    rows=list(csv.DictReader((OUT/'registration_metrics.csv').open()))
    points=np.load(OUT/'workspace_points.npz')
    sources=json.loads((OUT/'sources.json').read_text())
    fig=plt.figure(figsize=(12.8,7.1),facecolor='white')
    titles=['(a) Coarse Placement','(b) Legacy Registration','(c) Gaussian-Aware Registration\n(Ours)']
    image_axes=[]
    for i,key in enumerate('ABC'):
        ax=fig.add_axes([.022+i*.326,.490,.309,.435])
        ax.imshow(Image.open(OUT/'panels'/f'{key}_anchor.png'));ax.set_axis_off()
        ax.set_title(titles[i],loc='left',fontweight='bold',pad=9,fontsize=14)
        image_axes.append(ax)
    fig.text(.5,.459,'Cyan: shared M2 reference silhouette',ha='center',fontsize=11,color='#227882')
    positions=[[.047,.125,.173,.238],[.274,.125,.173,.238],[.501,.125,.173,.238]]
    keys=['silhouette_iou','reference_rgb_error','workspace_overflow']
    labels=['(d) Silhouette IoU ↑','(e) RGB Error ↓','(f) Overflow (%) ↓']
    for position,key,label in zip(positions,keys,labels):
        ax=fig.add_axes(position);metric(ax,rows,key,label,percent=key=='workspace_overflow')
        small,sa=plt.subplots(figsize=(3.4,2.8));metric(sa,rows,key,label,percent=key=='workspace_overflow')
        small.tight_layout();small.savefig(OUT/'panels'/f'{key}.png',dpi=300);plt.close(small)
    ax=fig.add_axes([.753,.150,.222,.245]);footprint(ax,points)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.84,.007),ncol=2,
               frameon=False,fontsize=10,columnspacing=1,handlelength=1.6)
    fig.savefig(OUT/'figure_F13.png',dpi=300,facecolor='white')
    fig.savefig(OUT/'figure_F13.pdf',facecolor='white')
    plt.close(fig)
    fp,fa=plt.subplots(figsize=(5,4.7));footprint(fa,points)
    fa.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False)
    fp.tight_layout();fp.savefig(OUT/'workspace_footprints.png',dpi=300)
    fp.savefig(OUT/'panels/workspace_footprint.png',dpi=300);plt.close(fp)
    caption=(
        'Fig. 13. Spatial registration of a Gaussian roller conveyor in an industrial scene (Case02). '
        '(a) Coarse placement from the saved M2 generation volume and support height, with no image fitting. '
        '(b) The original point-projection silhouette registration (alignment.fit, Powell optimization). '
        '(c) Gaussian-aware reference registration (fit_rasterized), independently initialized from its native yaw/size candidate pool without a historical transform seed. '
        'All three methods use the same frozen PBR-trained Gaussian asset, M2 camera, reference and generation volume, and original M1 background. '
        'Cyan contours show the same M2 reference silhouette; images use exactly the same camera and crop. '
        '(d-f) Unified evaluation of the exported Gaussian PLYs at 1024 x 1024: binary silhouette IoU (alpha threshold 0.5), '
        'mean absolute RGB error on the white-composited M2 reference foreground, and the percentage of all Gaussian centers outside the oriented M2 generation volume. '
        '(g) Top-down convex envelopes of the transformed Gaussian centers in the common M2 frame, together with the allowed workspace (dashed). '
        'A, B and C denote coarse, legacy and Gaussian-aware registration, respectively. '
        'The input asset, localization, generation space and background were not retrained or edited. '
        'Registration losses are method-specific; reported evaluation definitions are shared, and the RGB metric is not a claim of photometric ground truth.')
    (OUT/'caption.txt').write_text(caption+'\n')
    sources['figure']={'script':str(Path(__file__)),'resolution_png_dpi':300,
        'canvas_inches':[12.8,7.1],'pdf_text':'embedded vector fonts',
        'panels_abc_camera_and_crop_identical':True,'image_changes':'Common crop plus cyan reference contour only; no retouching.',
        'no_yaw_scale_heatmap':True}
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    (OUT/'README.txt').write_text(
        'Case02 only. Actual unmodified alignment.fit and fit_rasterized calls.\n'
        'A/B/C/m4 contain exported object_world.ply and transform.json.\n'
        'Each A/B/C/config.yaml records inputs and settings; m1-m3 are read-only source links.\n'
        'C has no existing_transform seed. No M1-M3 rerun or M4 background optimization.\n'
        'Primary overflow is all-Gaussian center count; covariance support overflow is an additional CSV diagnostic.\n'
        'Run experiment: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/run_f13_case02_registration.py --gpu 3\n'
        'Plot: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f13_case02_registration.py\n'
        'The experiment refuses to overwrite an existing output directory.\n')
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2000','-png',str(OUT/'figure_F13.pdf'),str(OUT/'pdf_preview')],check=True)
    print('F13 FIGURE COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
