"""Mechanism-oriented F14 layout from frozen Case00 candidates and real logs."""
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F14_occlusion_refinement_revised'
COLORS={'RGB-only':'#D48938','Full M4 Candidate':'#217D99'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13.5,
    'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
    'legend.fontsize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def read_csv(name):return list(csv.DictReader((OUT/name).open()))


def image_panel(ax,path,title=None):
    image=np.asarray(Image.open(path))
    if image.ndim == 2:
        ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    else:
        ax.imshow(image)
    ax.set_axis_off()
    if title:ax.set_title(title,pad=9,fontweight='bold')
    return image


def scene(ax,key,sources):
    image=image_panel(ax,OUT/'panels'/f'{key}_scene.png')
    main=np.array(sources['main_crop_xyxy_at1024']);detail=np.array(sources['detail_crop_xyxy_at1024'])
    scale=image.shape[1]/(main[2]-main[0]);xy=(detail[:2]-main[:2])*scale
    wh=(detail[2:]-detail[:2])*scale
    ax.add_patch(Rectangle(xy,*wh,fill=False,edgecolor='white',linewidth=1.15,linestyle=(0,(4,3))))


def detail(ax,key,sources):
    image_panel(ax,OUT/'panels'/f'{key}_base.png')
    p=np.array(sources['observation_arrow_pixel_detail'],float)
    ax.annotate('',xy=p,xytext=p+np.array([100.,-110.]),
        arrowprops=dict(arrowstyle='->',color='#00DDEB',lw=1.8,mutation_scale=14))


def opacity_map(ax,projection):
    mask=projection['initial_candidate_contribution']>.003
    values=np.ma.masked_where(~mask,projection['mean_opacity_reduction'])
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#F1F2F4')
    im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
    ax.set_axis_off();return im


def curves(ax,rows,value,scale):
    for name,color in COLORS.items():
        selected=[r for r in rows if r['method']==name]
        x=[int(r['step']) for r in selected];y=[float(r[value])*scale for r in selected]
        ax.plot(x,y,lw=2.0,color=color,label=name)
        ax.scatter([x[-1]],[y[-1]],color=color,s=23,zorder=4)
    ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000]);ax.set_xlabel('Iteration')
    ax.grid(color='#E5E8EC',linewidth=.7);ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y',useOffset=False,style='plain')


def main():
    sources=json.loads((OUT/'sources.json').read_text());guards=json.loads((OUT/'guard_status.json').read_text())
    visibility=read_csv('visibility_metrics.csv');environment=read_csv('environment_metrics.csv')
    projection=np.load(OUT/'occlusion_projection.npz')
    width,height=12.8,13.0
    fig=plt.figure(figsize=(width,height),facecolor='white')
    def axes(x,y,w,h):return fig.add_axes([x/width,y/height,w/width,h/height])
    starts=[.28,4.47,8.66];column_w=3.86
    titles=['(a) Initial Integration\nStep 0','(b) RGB-only Refinement\nStep 2000','(c) Full M4 Candidate\nStep 2000']
    base_metrics={r['method']:r['base_visibility'] for r in sources['local_visibility']}
    for key,x,title in zip('ABC',starts,titles):
        ax=axes(x,8.55,column_w,3.86);scene(ax,key,sources);ax.set_title(title,pad=10,fontweight='bold')
        zoom=axes(x,5.88,column_w,2.54);detail(zoom,key,sources)
        fig.text((x+column_w/2)/width,5.68/height,f'Base visibility: {base_metrics[key]*100:.1f}%',ha='center',fontsize=11.5)
    fig.text(.28/width,5.26/height,'(d) Dual-Mask Supervision',fontsize=13.5,fontweight='bold')
    fig.text(7.32/width,5.26/height,'(e) Occluding Gaussian Adjustment',fontsize=13.5,fontweight='bold')
    for x,name,title in [(0.28,'old_mask','Old Visible Mask'),(2.58,'new_mask','New Object Mask'),(4.88,'environment_mask','Environment Mask')]:
        ax=axes(x,3.08,2.04,1.94)
        image_panel(ax,OUT/'panels'/(name+'.png'));ax.set_title(title,fontsize=11,pad=7)
    fig.text(3.6/width,2.85/height,r'$M_{\rm env}=1-(M_{\rm old}\cup M_{\rm new})$',ha='center',fontsize=12)
    a=axes(7.32,3.43,2.46,1.62);image_panel(a,OUT/'panels/occluders_projection.png')
    a.set_title('Step 0 contribution',fontsize=11,pad=7)
    b=axes(10.03,3.43,2.46,1.62);im=opacity_map(b,projection)
    b.set_title('Opacity reduction',fontsize=11,pad=7)
    cbax=axes(10.13,3.16,2.25,.11)
    cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,.5,1]);cb.ax.tick_params(labelsize=9.5)
    fig.text(9.85/width,2.85/height,r'313 candidates; $\alpha_0-\alpha_{2000}$',ha='center',fontsize=11)
    fig.text(.72/width,2.48/height,'(f) Trusted Object Visibility',fontsize=13.5,fontweight='bold')
    fig.text(7.04/width,2.48/height,'(g) Environment Preservation',fontsize=13.5,fontweight='bold')
    ax=axes(.72,.63,5.03,1.59);curves(ax,visibility,'trusted_visibility',100.)
    ax.set_ylabel('Trusted visibility (%)');ax.yaxis.set_major_locator(MultipleLocator(1))
    ax=axes(7.04,.63,5.03,1.59);curves(ax,environment,'heldout_environment_rgb_l1',1000.)
    ax.set_ylabel(r'Held-out RGB L1 ($\times 10^{-3}$)');ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.legend(handles=[Line2D([],[],color=color,lw=2,label=name) for name,color in COLORS.items()],
        loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,columnspacing=2.5)
    fig.savefig(OUT/'figure_F14_revised.png',dpi=300,facecolor='white')
    fig.savefig(OUT/'figure_F14_revised.pdf',facecolor='white')
    plt.close(fig)

    # Independent complete panels retain the same real pixels and definitions.
    for key,title in zip('ABC',titles):
        small=plt.figure(figsize=(4.5,7.3),facecolor='white')
        ax=small.add_axes([.03,.47,.94,.49]);scene(ax,key,sources);ax.set_title(title,fontweight='bold')
        ax=small.add_axes([.03,.065,.94,.37]);detail(ax,key,sources)
        small.text(.5,.025,f'Base visibility: {base_metrics[key]*100:.1f}%',ha='center')
        small.savefig(OUT/'panels'/f'{key}_scene_and_detail.png',dpi=300);plt.close(small)
    small,aa=plt.subplots(1,3,figsize=(8,3.1))
    for ax,name,title in zip(aa,['old_mask','new_mask','environment_mask'],['Old Visible Mask','New Object Mask','Environment Mask']):
        image_panel(ax,OUT/'panels'/(name+'.png'));ax.set_title(title,fontsize=12)
    small.tight_layout();small.savefig(OUT/'panels/dual_mask_supervision.png',dpi=300);plt.close(small)
    small=plt.figure(figsize=(8,3.6),facecolor='white')
    aa=small.add_axes([.025,.16,.455,.73]);image_panel(aa,OUT/'panels/occluders_projection.png');aa.set_title('Occluding Gaussians: Step 0',fontsize=12)
    bb=small.add_axes([.52,.16,.455,.73]);im=opacity_map(bb,projection);bb.set_title('Contribution-weighted opacity reduction',fontsize=11)
    cbax=small.add_axes([.59,.075,.30,.035]);cb=small.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    cb.set_label(r'$\alpha_0-\alpha_{2000}$',fontsize=11);cb.ax.tick_params(labelsize=10)
    small.savefig(OUT/'occlusion_gaussian_analysis.png',dpi=300);small.savefig(OUT/'panels/occlusion_gaussian_analysis.png',dpi=300);plt.close(small)
    for filename,rows,metric,scale,title,ylabel in [
        ('trusted_visibility',visibility,'trusted_visibility',100.,'(f) Trusted Object Visibility','Trusted visibility (%)'),
        ('environment_preservation',environment,'heldout_environment_rgb_l1',1000.,'(g) Environment Preservation',r'Held-out RGB L1 ($\times10^{-3}$)')]:
        small,ax=plt.subplots(figsize=(6,3.3));curves(ax,rows,metric,scale);ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylabel(ylabel);ax.legend(frameon=False,fontsize=10);small.tight_layout()
        small.savefig(OUT/'panels'/(filename+'.png'),dpi=300);plt.close(small)

    caption=(
        'Fig. 14. Dual-mask-guided refinement of residual Gaussian occlusion around the base of the fixed Case00 industrial robot. '
        '(a-c) The initial integration, the existing RGB-only 2,000-step control, and the existing full M4 2,000-step candidate '
        'rendered with identical anchor-camera geometry and common full-device and base-detail crops. Dashed boxes locate the detail region; '
        'cyan arrows mark the same pixel selected by the largest initial occluder contribution within the independently rendered base. '
        'Base visibility is the composited contribution of the lowest 20% of device Gaussian centers divided by their independent contribution, '
        'measured using the unchanged native anchor camera: 52.2%, 52.9% and 93.6%, respectively. '
        '(d) The actual cached anchor-view old-visible and new-object masks, and their union complement. The masks illustrate the geometric '
        'supervision partition; real-photo RGB supervision is applied to reliable, edge-protected local pixels in registered photographic views, '
        'not to the synthetic anchor. '
        '(e) The actual original compositing contribution of the 313 occlusion-optimization candidates, highlighted in red, and the '
        'contribution-weighted mean decrease in their opacity. Only existing Gaussian opacity is adjusted; the candidates are not deleted. '
        'The map is calculated by rasterizing per-Gaussian opacity changes with the original scene compositing weights, not by editing the RGB image. '
        '(f-g) Unsmoothened held-out-view evaluation logs, sampled every 50 steps from 0 to 2,000, averaged over the same ten views: '
        'trusted device contribution and local environment RGB L1 against registered photographs. '
        'The fixed object, placement, original background, supervision cache, learning rates and sampling seed are shared. '
        'The existing RGB-only control disables visibility, occlusion-specific environment/legal-foreground loss terms and occlusion regularization, '
        'while retaining environment RGB and color/opacity parameter regularization. '
        'The full candidate increases trusted held-out visibility from 0.648 to 0.696 (RGB-only: 0.650), but this is not an accepted final-scene improvement. '
        'Both methods fail the unchanged per-view preservation guards; no evaluated nonzero step is accepted, and both guarded exports retain Step 0. '
        'For the full candidate, the largest outside-union and legal-foreground MAEs are 0.01740 and 0.01316, respectively, versus the 0.005 limits. '
        'The held-out mean environment L1 alone passes its guard, demonstrating why mean curves do not replace per-view checks. '
        'No new localization, generation, device training or background optimization was performed for this figure.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    sources['figure']={'script':str(Path(__file__)),'canvas_inches':[width,height],'png_dpi':300,
        'layout':'Three columns of scene+base detail; masks and real opacity attribution; two held-out curves.',
        'image_annotations':'White crop box and the same cyan observation arrow; image pixels otherwise unretouched.',
        'opacity_map_threshold_for_display':.003,'opacity_map_color_range':[0,1],
        'curves_unsmoothed':True,'candidate_not_guarded_final':True,'no_guard_threshold_changes':True}
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    (OUT/'README.txt').write_text(
        'F14 Case00 residual-occlusion mechanism, existing 2000-step candidates.\n'
        'Read caption.txt for exact candidate/guard interpretation and the RGB-only control.\n'
        'guard_status.json records all failing views and unchanged thresholds.\n'
        'visibility_metrics.csv / environment_metrics.csv: 41 evaluations per method on ten common held-out cameras.\n'
        'anchor_base_visibility.csv: same-camera base visibility. occlusion_gaussians.csv: all 313 actual selected Gaussian opacity changes.\n'
        'occlusion_projection.npz: actual compositing attribution and opacity-change maps. records/: original logs and indices.\n'
        'No methods were trained or rerun. Original experiments and F14 remain unchanged.\n'
        'Extract/render: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/extract_f14_occlusion.py --gpu 3\n'
        'Plot: envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f14_occlusion.py\n')
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2200','-png',str(OUT/'figure_F14_revised.pdf'),str(OUT/'pdf_preview')],check=True)
    print('F14 REVISED FIGURE COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
