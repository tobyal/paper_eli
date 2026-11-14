"""Append a measured yaw/scale objective slice to the accepted six-panel F13."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_f13_case02_final as prior

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F13_case02_registration'
ACCEPTED=SRC.parent/'F13_case02_registration_final'
OUT=SRC.parent/'F13_case02_registration_with_heatmap'


def edges(values):
    values=np.asarray(values)
    return np.r_[values[0]-(values[1]-values[0])/2,(values[:-1]+values[1:])/2,
                 values[-1]+(values[-1]-values[-2])/2]


def heatmap(ax,fig,cax,grid,summary):
    selected=summary['selected_solution'];minimum=summary['slice_minimum']
    values=grid['objective']
    mesh=ax.pcolormesh(edges(grid['yaw_degrees']),edges(grid['scale_over_selected']),values,
        cmap='viridis',shading='flat',antialiased=False,rasterized=True,
        vmin=float(values.min()),vmax=float(values.max()))
    # Both coordinates are their measured values, even if markers coincide.
    ax.scatter([minimum['yaw_degrees']],[minimum['scale_over_selected']],s=320,
               marker='o',facecolors='none',edgecolors='#FF724A',linewidths=2.0,
               zorder=6,label='Slice Minimum')
    ax.scatter([selected['yaw_degrees']],[1.0],s=170,marker='*',c='white',
               edgecolors='#222222',linewidths=1.1,zorder=7,label='Selected Solution')
    ax.set_xlim(edges(grid['yaw_degrees'])[[0,-1]])
    ax.set_ylim(edges(grid['scale_over_selected'])[[0,-1]])
    ax.set_xlabel('Yaw (degrees)',fontsize=12)
    ax.set_ylabel('Scale / Selected Scale',fontsize=12)
    ax.set_yticks(np.arange(.7,1.061,.05))
    ax.tick_params(labelsize=11)
    colorbar=fig.colorbar(mesh,cax=cax)
    colorbar.set_label('Registration Objective',fontsize=12,labelpad=10)
    colorbar.ax.tick_params(labelsize=10.5)
    return mesh


def main():
    summary=json.loads((OUT/'grid_summary.json').read_text())
    grid=np.load(OUT/'yaw_scale_grid.npz')
    before=json.loads((ACCEPTED/'sources.json').read_text())
    records=before['footprint_geometry'];points=np.load(SRC/'workspace_points.npz')
    rows=list(csv.DictReader((SRC/'registration_metrics.csv').open()))
    for p in (ACCEPTED/'panels').iterdir():
        if p.is_file():shutil.copy2(p,OUT/'panels'/p.name)

    # Keep the first two rows at their accepted physical size, then add a third.
    width,top_h,extra_h=12.8,8.3,4.1
    total_h=top_h+extra_h
    fig=plt.figure(figsize=(width,total_h),facecolor='white')
    def above(box):
        x,y,w,h=box
        return [x,(extra_h+y*top_h)/total_h,w,h*top_h/total_h]
    def top_text(x,y,text,**kw):
        return fig.text(x,(extra_h+y*top_h)/total_h,text,**kw)
    titles=['(a) Coarse Placement','(b) Legacy Registration','(c) Gaussian-Aware Registration\n(Ours)']
    for i,(key,title) in enumerate(zip('ABC',titles)):
        ax=fig.add_axes(above([.025+i*.326,.510,.312,.420]))
        ax.imshow(Image.open(ACCEPTED/'panels'/f'{key}_anchor.png'));ax.set_axis_off()
        ax.set_title(title,loc='left',pad=9,fontweight='bold')
    a=fig.add_axes(above([.063,.115,.244,.330]));prior.metric(a,rows,'silhouette_iou')
    b=fig.add_axes(above([.378,.115,.244,.330]));prior.metric(b,rows,'reference_rgb_error')
    f=fig.add_axes(above([.693,.115,.282,.330]));prior.footprint(f,points,records)
    for x,label in [(.063,'(d) Silhouette IoU ↑'),(.378,'(e) RGB Error ↓'),(.674,'(f) Workspace Footprint Comparison')]:
        top_text(x,.471,label,fontweight='bold',fontsize=13.5)
    fig.legend(handles=prior.legend_handles(),loc='lower center',
        bbox_to_anchor=(.83,(extra_h+.009*top_h)/total_h),frameon=False,ncol=4,
        columnspacing=.8,handlelength=1.2,handletextpad=.4,fontsize=10.5)

    # Flat cells represent actual samples; there is no smoothed/interpolated surface.
    g=fig.add_axes([.105,.55/total_h,.705,2.85/total_h])
    colorbar=fig.add_axes([.829,.55/total_h,.017,2.85/total_h])
    heatmap(g,fig,colorbar,grid,summary)
    fig.text(.063,3.76/total_h,'(g) Yaw–Scale Registration Objective',fontweight='bold',fontsize=13.5)
    handles,labels=g.get_legend_handles_labels()
    fig.legend([handles[1],handles[0]],[labels[1],labels[0]],loc='lower right',
        bbox_to_anchor=(.905,3.55/total_h),ncol=2,frameon=False,fontsize=11,
        columnspacing=1.4,handletextpad=.6)
    fig.savefig(OUT/'figure_F13_final.png',dpi=300,facecolor='white')
    fig.savefig(OUT/'figure_F13_final.pdf',facecolor='white')
    plt.close(fig)

    sf=plt.figure(figsize=(10.5,4.5),facecolor='white')
    ax=sf.add_axes([.105,.15,.715,.69]);cb=sf.add_axes([.845,.15,.02,.69])
    heatmap(ax,sf,cb,grid,summary)
    sf.text(.035,.94,'(g) Yaw–Scale Registration Objective',fontweight='bold',fontsize=13.5)
    handles,labels=ax.get_legend_handles_labels()
    sf.legend([handles[1],handles[0]],[labels[1],labels[0]],loc='lower right',
              bbox_to_anchor=(.94,.85),ncol=2,frameon=False,fontsize=10.5)
    sf.savefig(OUT/'yaw_scale_objective.png',dpi=300)
    sf.savefig(OUT/'panels/yaw_scale_objective.png',dpi=300);plt.close(sf)

    selected=summary['selected_solution'];minimum=summary['slice_minimum']
    outcome=('The selected solution and the sampled slice minimum coincide; both markers are plotted at their true shared coordinate.'
             if summary['selected_is_grid_minimum'] else
             'The selected solution and the sampled slice minimum differ; both markers retain their measured coordinates.')
    caption=(ACCEPTED/'caption.txt').read_text().strip()+' '+(
        '(g) A yaw-scale slice of the complete Gaussian-aware registration objective, evaluated by the unchanged '
        'registration code with the saved horizontal support-center offsets fixed. All 2,257 samples '
        '(61 yaw values at 1.5-degree spacing and 37 scale ratios at 0.01 spacing) use the original fine-stage '
        '640 x 640 Gaussian rasterizer and include silhouette, RGB-structure, boundary, center and spatial-volume terms. '
        'The yaw range is the selected yaw +/-45 degrees; scale ratios span 0.70-1.06 within the native parameter bounds. '
        'Flat colored cells show directly computed losses using linear Viridis normalization, without interpolation or smoothing. '
        'The white star denotes the saved selected solution (scale ratio 1.0); the orange open circle denotes the '
        'minimum among the sampled cells. '+outcome+' '
        'This is a conditional objective slice, not a new optimization run or an optimization trajectory. '
        'The existing A/B/C registrations and the 1024 x 1024 evaluation metrics in (d-e) are unchanged.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    source={
        'case':'case02','accepted_six_panel_source':str(ACCEPTED),
        'accepted_source_sha256':{str(ACCEPTED/n):prior.sha(ACCEPTED/n) for n in ['figure_F13_final.png','figure_F13_final.pdf','sources.json','caption.txt']},
        'prior_panels_and_metrics':before,'heatmap_computation':summary,
        'grid_csv':str(OUT/'yaw_scale_objective.csv'),'grid_npz':str(OUT/'yaw_scale_grid.npz'),
        'grid_csv_sha256':prior.sha(OUT/'yaw_scale_objective.csv'),'grid_npz_sha256':prior.sha(OUT/'yaw_scale_grid.npz'),
        'display':{'colormap':'viridis','normalization':'linear over all actual sampled losses',
            'minimum_color_value':float(grid['objective'].min()),'maximum_color_value':float(grid['objective'].max()),
            'interpolation':False,'selected_marker':selected,'slice_minimum_marker':minimum,
            'same_fixed_translation_for_both_markers':True,'canvas_inches':[width,total_h],'dpi':300},
        'scene_panels_pixel_identical':{k:prior.sha(OUT/'panels'/f'{k}_anchor.png')==prior.sha(ACCEPTED/'panels'/f'{k}_anchor.png') for k in 'ABC'},
        'no_registration_training_or_asset_edits':True,'script':str(Path(__file__)),
        'script_sha256':prior.sha(__file__)}
    assert all(source['scene_panels_pixel_identical'].values())
    (OUT/'sources.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    shutil.copy2(Path(prior.__file__),OUT/'code'/Path(prior.__file__).name)
    (OUT/'README.txt').write_text(
        'F13 Case02, accepted six panels plus measured yaw-scale objective slice.\n'
        'Compute (server): envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/compute_f13_yaw_scale.py --gpu 2\n'
        'Layout: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f13_with_heatmap.py\n'
        'The compute script refuses to overwrite an existing directory. The layout script only writes this new figure directory.\n'
        'The original accepted F13 and experiment figure/F13 remain unchanged.\n'
        'yaw_scale_objective.csv has every real sample and each loss term; yaw_scale_grid.npz preserves the matrix; grid_summary.json records fixed parameters, source hashes and both markers.\n'+outcome+'\n')
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2200','-png',str(OUT/'figure_F13_final.pdf'),str(OUT/'pdf_preview')],check=True)
    print('F13 HEATMAP FIGURE COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
