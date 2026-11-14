import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon,Rectangle
from figure_common import *

def main():
    d=folder('F13_spatial_alignment');results=read(d/'ablation_results.json');surface=read(d/'objective_surface.json')
    fig=plt.figure(figsize=(9,8.3));gs=fig.add_gridspec(3,6,height_ratios=[1.3,1,1.35],left=.075,right=.98,bottom=.10,top=.91,hspace=.58,wspace=.8)
    fig.suptitle('Spatial Registration and Workspace-Constrained Placement',y=.975,fontsize=13,fontweight='bold');axes=[]
    for j,(method,label) in enumerate([('A_bbox','A: Bounding-box placement'),('B_silhouette','B: Silhouette only'),('C_full','C: Full registration')]):
        ax=fig.add_subplot(gs[0,j*2:j*2+2]);axes.append(ax);ax.imshow(Image.open(d/'panels'/f'case02_{method}_contour.png'));ax.axis('off');ax.set_title(('(a) ' if j==0 else '')+label,fontsize=9)
    for j,(key,label,mult) in enumerate([('mask_iou','Silhouette IoU',1),('center_offset_fraction','Support-center offset / footprint',1),('support_outside_fraction','Gaussian support outside (%)',100)]):
        ax=fig.add_subplot(gs[1,j*2:j*2+2]);axes.append(ax)
        for i,case in enumerate(['case00','case02']):
            vals=[next(x for x in results if x['case']==case and x['method']==m)[key]*mult for m in ['A_bbox','B_silhouette','C_full']]
            ax.bar(np.arange(3)+(i-.5)*.34,vals,.34,color=COLORS[i],label=case)
        ax.set_xticks(range(3),['BBox','Silhouette','Full']);ax.set_title('(b) '+label,fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if j==0:ax.legend(frameon=False)
    ax=fig.add_subplot(gs[2,:4]);axes.append(ax);s=surface['values']
    yaw=sorted(set(x['yaw_degrees'] for x in s));scale=sorted(set(x['scale_factor'] for x in s))
    z=np.array([x['objective'] for x in s]).reshape(len(yaw),len(scale)).T
    img=ax.pcolormesh(yaw,scale,z,shading='nearest',cmap='viridis');fig.colorbar(img,ax=ax,label='Full registration objective',pad=.02,fraction=.04)
    selected=surface['selected'];yy=np.degrees(selected['parameters_radians_logscale_dx_dz'][0])
    ax.scatter([yy],[1],marker='*',s=140,c='white',edgecolors='black',label='Saved solution')
    ax.set(title='(c) Case02 yaw-scale objective slice',xlabel='Yaw (degrees)',ylabel='Scale / selected scale');ax.legend(frameon=False,loc='upper right')
    ax=fig.add_subplot(gs[2,4:]);axes.append(ax);pts=np.load(d/'workspace_points.npz');lo=pts['low'];hi=pts['high']
    ax.add_patch(Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],fill=False,color='black',ls='--',lw=1.5,label='Allowed space'))
    for key,col,label in [('old',COLORS[4],'Old equipment'),('new',COLORS[0],'New equipment')]:
        points=pts[key][:,[0,2]];low,high=np.quantile(points,[.005,.995],axis=0);points=points[((points>=low)&(points<=high)).all(1)]
        hull=points[ConvexHull(points).vertices];ax.add_patch(Polygon(hull,alpha=.28,color=col,label=label))
    ax.autoscale();ax.set_aspect('equal');ax.set(title='(d) Case02 workspace footprint',xlabel='Local x',ylabel='Local z');ax.legend(frameon=False,fontsize=7)
    fig.text(.075,.025,'Fixed assets and support plane. Offset is relative to the designated support center, not pose ground truth.\nOutside fraction: opacity-weighted sampled Gaussian covariance supports crossing the allowed box.',fontsize=8)
    rows=[dict(record='registration',**{k:v for k,v in x.items() if isinstance(v,(str,int,float,bool))}) for x in results]
    rows += [dict(record='objective_slice',case='case02',**x) for x in s]
    caption='Figure 13. Registration ablation with fixed PBR-trained Gaussian assets. (a) Case02 scene insertion under bounding-box initialization, silhouette-only optimization, and the saved full registration; the red contour is the same Qwen reference mask. (b) Silhouette IoU, normalized support-center displacement, and opacity-weighted support-outside fraction for Case00 and Case02. Both optimization methods use real Gaussian rasterization and the same recorded initialization pool; the silhouette ablation removes RGB, center and volume losses while retaining common bounded pose parameters and the fixed support plane. Center displacement is a workspace consistency measure, not ground-truth pose error. (c) Directly rendered objective samples at fixed selected horizontal translation; this is a slice, not an independently optimized solution at each grid cell. (d) Robust Gaussian-center footprint hulls and the allowed workspace. No background optimization or asset generation was performed for this ablation.'
    export(fig,d,'F13',caption,dict(experiments=[read(d/(x+'_provenance.json')) for x in ['case00','case02']],surface=source(d/'objective_surface.json'),existing_full_results=True),rows,axes)

if __name__=='__main__':main()
