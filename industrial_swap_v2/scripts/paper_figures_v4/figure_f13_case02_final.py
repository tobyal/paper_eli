"""Layout-only F13 revision; consume saved A/B/C data, never run a solver."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration'
OUT = SRC.parent/'F13_case02_registration_final'
COLORS = dict(A='#7B8490', B='#D48938', C='#217D99')
NAMES = dict(A='Coarse Placement', B='Legacy Registration', C='Gaussian-Aware Registration')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
    'axes.titlesize':13.5, 'axes.labelsize':12, 'xtick.labelsize':11.5,
    'ytick.labelsize':11.5, 'legend.fontsize':10.5, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':300})


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def geometry(points, transforms):
    # Read the unchanged original asset to define one shared intrinsic axis.
    source=Path(transforms['C']['source_local'])
    vertex=PlyData.read(str(source))['vertex'].data
    local=np.column_stack([vertex[k] for k in ('x','y','z')]).astype(np.float64)
    center=local.mean(0)
    eigenvalues,eigenvectors=np.linalg.eigh(np.cov(local[:,[0,2]].T))
    direction=eigenvectors[:,-1]
    direction*=np.sign(direction[np.argmax(np.abs(direction))])
    axis=np.array([direction[0],0.,direction[1]])
    origin,axes=points['origin'],points['axes']
    records={}
    for key in 'ABC':
        t=transforms[key];rotation=np.asarray(t['rotation']);translation=np.asarray(t['translation'])
        world_center=rotation@center*t['scale']+translation
        center_m2=(world_center-origin)@axes
        axis_m2=axes.T@rotation@axis
        horizontal=axis_m2[[0,2]];horizontal/=np.linalg.norm(horizontal)
        actual=points[key].astype(np.float64)
        assert np.allclose(actual.mean(0),center_m2,atol=2e-6)
        xz=actual[:,[0,2]];hull=ConvexHull(xz)
        records[key]={'center_world':world_center.tolist(),'center_m2':center_m2.tolist(),
            'direction_m2_xz':horizontal.tolist(),'footprint_hull_m2_xz':xz[hull.vertices].tolist(),
            'gaussian_count':len(actual)}
    return records,{'asset':str(source.resolve()),'asset_sha256':sha(source),
        'center_definition':'Arithmetic mean of all 200000 original Gaussian centers, transformed into the common M2 frame.',
        'axis_definition':'Principal horizontal axis of the original Gaussian-center distribution; sign fixed by making the largest component positive, then rotated with each actual transform. Indicates geometric orientation, not motor transport direction.',
        'local_center':center.tolist(),'local_axis':axis.tolist(),
        'horizontal_eigenvalues':eigenvalues.tolist(),
        'outline_definition':'Convex hull of all saved transformed Gaussian centers in the M2 x-z plane; no outlier pruning or manual footprint adjustment.'}


def metric(ax,rows,key):
    vals=[float(row[key]) for row in rows]
    ax.bar(range(3),vals,width=.57,color=list(COLORS.values()),zorder=3)
    for i,v in enumerate(vals):
        ax.annotate(f'{v:.3f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=13)
    ax.set_xticks(range(3),list('ABC'))
    ax.set_ylim(0,1.03 if key=='silhouette_iou' else .60)
    ax.yaxis.set_major_locator(MultipleLocator(.2 if key=='silhouette_iou' else .1))
    ax.set_axisbelow(True);ax.yaxis.grid(color='#E5E8EC',linewidth=.8)
    ax.spines['left'].set_color('#72777E');ax.spines['bottom'].set_color('#72777E')


def footprint(ax,points,records):
    low,high=points['low'],points['high']
    for key,color in COLORS.items():
        record=records[key];outline=np.array(record['footprint_hull_m2_xz'])
        ax.add_patch(Polygon(outline,closed=True,facecolor=color,alpha=.15,edgecolor='none',zorder=2))
        ax.add_patch(Polygon(outline,closed=True,fill=False,edgecolor=color,linewidth=2.1,zorder=3))
    ax.add_patch(Rectangle((low[0],low[2]),high[0]-low[0],high[2]-low[2],
        fill=False,edgecolor='#25292F',linewidth=1.8,linestyle=(0,(5,3)),zorder=4))
    length=.18*np.linalg.norm((high-low)[[0,2]])
    for key,color in COLORS.items():
        center=np.array(records[key]['center_m2'])[[0,2]]
        direction=np.array(records[key]['direction_m2_xz'])
        end=center+length*direction
        ax.annotate('',xy=end,xytext=center,arrowprops=dict(arrowstyle='-|>',lw=2.2,
            color=color,mutation_scale=15,shrinkA=0,shrinkB=0),zorder=6)
        ax.plot(*center,'o',color=color,markersize=6.5,markeredgecolor='white',markeredgewidth=.9,zorder=7)
        label=end+.035*direction
        ax.text(*label,key,color=color,fontweight='bold',ha='center',va='center',fontsize=12,zorder=8)
    allp=np.concatenate([points[k][:,[0,2]] for k in 'ABC']+
                        [np.array([[low[0],low[2]],[high[0],high[2]]])])
    # Square limits and equal aspect give the same spatial unit length on x/z.
    limits=[np.floor((allp.min()-.04)/.1)*.1,np.ceil((allp.max()+.04)/.1)*.1]
    ax.set_xlim(limits);ax.set_ylim(limits);ax.set_aspect('equal',adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(.2));ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.set_xlabel('Local x',labelpad=4);ax.set_ylabel('Local z',labelpad=4)
    ax.set_axisbelow(True);ax.grid(color='#E4E7EB',linewidth=.75)
    ax.spines['left'].set_color('#72777E');ax.spines['bottom'].set_color('#72777E')
    return limits,length


def legend_handles():
    return [Line2D([],[],color='#25292F',ls='--',lw=1.8,label='Allowed Workspace')]+[
        Patch(facecolor=color,edgecolor=color,alpha=.7,label=key) for key,color in COLORS.items()]


def main():
    if OUT.exists() and '--refresh-layout' not in sys.argv: raise FileExistsError('Keep prior outputs: '+str(OUT))
    OUT.mkdir(exist_ok=True);panels=OUT/'panels';panels.mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
    files=[SRC/'registration_metrics.csv',SRC/'registration_transforms.json',SRC/'workspace_points.npz',SRC/'sources.json']
    files += [SRC/'panels'/f'{key}_anchor.png' for key in 'ABC']
    before={str(p):sha(p) for p in files}
    rows=list(csv.DictReader((SRC/'registration_metrics.csv').open()))
    transforms=json.loads((SRC/'registration_transforms.json').read_text())
    previous=json.loads((SRC/'sources.json').read_text())
    points=np.load(SRC/'workspace_points.npz')
    records,geometry_source=geometry(points,transforms)
    fig=plt.figure(figsize=(12.8,8.3),facecolor='white')
    image_positions=[[.025+i*.326,.510,.312,.420] for i in range(3)]
    titles=['(a) Coarse Placement','(b) Legacy Registration','(c) Gaussian-Aware Registration\n(Ours)']
    for key,title,position in zip('ABC',titles,image_positions):
        src=SRC/'panels'/f'{key}_anchor.png'
        shutil.copy2(src,panels/f'{key}_anchor.png')
        ax=fig.add_axes(position);ax.imshow(Image.open(src));ax.set_axis_off()
        ax.set_title(title,loc='left',pad=9,fontweight='bold')
    bottom_y,bottom_h=.115,.330
    a=fig.add_axes([.063,bottom_y,.244,bottom_h]);metric(a,rows,'silhouette_iou')
    b=fig.add_axes([.378,bottom_y,.244,bottom_h]);metric(b,rows,'reference_rgb_error')
    f=fig.add_axes([.693,bottom_y,.282,bottom_h]);limits,arrow_length=footprint(f,points,records)
    # Titles use common figure baselines, independent of equal-aspect adjustment.
    fig.text(.063,.471,'(d) Silhouette IoU ↑',fontweight='bold',fontsize=13.5)
    fig.text(.378,.471,'(e) RGB Error ↓',fontweight='bold',fontsize=13.5)
    fig.text(.674,.471,'(f) Workspace Footprint Comparison',fontweight='bold',fontsize=13.5)
    fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.83,.009),
               frameon=False,ncol=4,columnspacing=.8,handlelength=1.2,handletextpad=.4,fontsize=10.5)
    fig.savefig(OUT/'figure_F13_final.png',dpi=300,facecolor='white')
    fig.savefig(OUT/'figure_F13_final.pdf',facecolor='white')
    plt.close(fig)
    for key,title in [('silhouette_iou','(d) Silhouette IoU ↑'),('reference_rgb_error','(e) RGB Error ↓')]:
        sf,ax=plt.subplots(figsize=(4.2,3.7));metric(ax,rows,key)
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        sf.tight_layout();sf.savefig(panels/f'{key}.png',dpi=300);plt.close(sf)
    sf,ax=plt.subplots(figsize=(5.5,5.5));footprint(ax,points,records)
    ax.set_title('(f) Workspace Footprint Comparison',loc='left',fontweight='bold',pad=12)
    ax.legend(handles=legend_handles(),loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
    sf.subplots_adjust(left=.15,right=.95,top=.90,bottom=.23)
    sf.savefig(OUT/'workspace_footprint.png',dpi=300)
    sf.savefig(panels/'workspace_footprint.png',dpi=300);plt.close(sf)
    caption=(
        'Fig. 13. Spatial registration of a Gaussian roller conveyor in an industrial scene (Case02). '
        '(a) Coarse placement from the M2 generation volume and support height. '
        '(b) Legacy point-projection silhouette registration using Powell optimization. '
        '(c) Gaussian-aware reference registration using real Gaussian rasterization, reference silhouette/RGB structure and spatial constraints. '
        'All methods use the same frozen PBR-trained asset, M2 anchor camera, reference mask, generation space and original M1 background; '
        'the scene images have identical camera, crop and size. Cyan denotes the shared M2 reference silhouette. '
        '(d) Binary silhouette IoU at an alpha threshold of 0.5. (e) Mean absolute RGB error on the white-composited reference foreground. '
        'Both metrics use the saved unified 1024 x 1024 evaluation without re-running registration. '
        '(f) Actual horizontal convex envelopes of all transformed Gaussian centers in the common M2 local frame. '
        'The black dashed rectangle indicates the allowed workspace; gray, orange and blue-green indicate A, B and C. '
        'Dots mark Gaussian-center centroids, and arrows show the same asset-local geometric long axis transformed by each pose. '
        'Arrow length is standardized for readability and does not denote device length or conveyor travel direction. '
        'Equal axis scaling preserves footprint geometry. No footprint, scene, transform or metric was manually adjusted; '
        'the complete original evaluation remains in the source experiment.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    sources={'case':'case02','source_experiment':str(SRC),'source_sha256':before,
        'camera':previous['camera'],'evaluation_camera':previous['evaluation_camera'],
        'shared_crop_xyxy':previous['crop_xyxy'],
        'rendered_panels_copied_without_modification':{k:sha(panels/f'{k}_anchor.png')==before[str(SRC/'panels'/f'{k}_anchor.png')] for k in 'ABC'},
        'displayed_metrics':[{k:r[k] for k in ['method','name','silhouette_iou','reference_rgb_error']} for r in rows],
        'complete_original_evaluation':str(SRC/'registration_metrics.csv'),
        'footprint_geometry':records,'geometry_source':geometry_source,
        'axis_limits_x_and_z':limits,'equal_aspect':True,'direction_arrow_length_scene_units':float(arrow_length),
        'solver_or_model_rerun':False,'Gaussian_rerender':False,'source_files_unchanged':all(sha(p)==h for p,h in before.items()),
        'layout':{'rows':2,'panels':6,'canvas_inches':[12.8,8.3],'png_dpi':300,'pdf_fonts':'embedded vector text',
                  'removed_display_panel':'Overflow only; complete source records preserved'},
        'script':str(Path(__file__)),'script_sha256':sha(__file__)}
    assert sources['source_files_unchanged'] and all(sources['rendered_panels_copied_without_modification'].values())
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2000','-png',str(OUT/'figure_F13_final.pdf'),str(OUT/'pdf_preview')],check=True)
    print('F13 FINAL LAYOUT COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
