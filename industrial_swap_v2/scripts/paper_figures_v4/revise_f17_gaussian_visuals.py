"""F17 a/b: real Gaussian splats, continuous opacity display and isolated insets.

No localization or evaluation is recomputed. Original indices/covariances/model
files are immutable. A common display-only opacity exposure is documented, and
unmodified-opacity contribution is checked against the archived evaluation mask.
"""
import argparse, hashlib, html, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation, binary_erosion

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F17_localization_comparison'
METHODS=['GaussianEditor','DGE','Ours']
QUAL=['M1-03_frame_000211','M1-05_frame_000228']
SCALE=3
EXPOSURE=.12
OVERLAY=.80

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def protected():
    files=[OUT/f for f in ['figure_F17.png','figure_F17.pdf','metrics.json','iou_per_view.csv','iou_per_target.csv','iou_summary.csv','sources.json','caption.txt','panels/c_statistics.png']]
    for folder in ['selection_indices','ground_truth_masks','projected_masks']:
        files.extend(p for p in (OUT/folder).rglob('*') if p.is_file())
    return {str(p.relative_to(OUT)):sha(p) for p in files}
def crop_from_gt(mask):
    yy,xx=np.where(mask);h,w=mask.shape
    cw=max(260,(xx.max()-xx.min()+1)*2.2,(yy.max()-yy.min()+1)*1.65*4/3)
    cw=min(cw,w,h*4/3);ch=cw*3/4
    x0=min(max((xx.min()+xx.max())/2-cw/2,0),w-cw)
    y0=min(max((yy.min()+yy.max())/2-ch/2,0),h-ch)
    return tuple(round(v) for v in [x0,y0,x0+cw,y0+ch])
def rgb_image(a):return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
def contour(a,gt):
    a=a.copy();edge=binary_dilation(gt,iterations=2*SCALE)^binary_erosion(gt,iterations=SCALE)
    a[edge]=[0,210/255,230/255];return a

def render(gpu):
    os.environ.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=str(gpu),GSLE_OUTPUT=str(OUT/'visualization_adapter'),GSLE_RECORDS=str(OUT/'records_revised'),SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(ROOT/'gsle'))
    import torch
    import common as c
    panels=OUT/'panels_revised';maps=OUT/'projected_masks_revised'
    panels.mkdir(exist_ok=True);maps.mkdir(exist_ok=True)
    snapshot=protected();src=read(OUT/'sources.json');scene_path=Path(src['source_scene'])
    assert sha(scene_path)==src['scene_sha256']
    scene=c.GaussianData.load(scene_path);views={v['task_id']:v for v in read(OUT/'evaluation_views.json')['views']}
    records=[];crops={}
    for tid in QUAL:
        label=tid.split('_')[0];cam=c.Camera.from_record(views[tid]['camera']);hi=cam.resize(cam.width*SCALE,cam.height*SCALE)
        photo=Image.open(OUT/'photos'/f'{tid}.png').convert('RGB');gt=Image.open(OUT/'ground_truth_masks'/tid/'mask.png')
        crop=crop_from_gt(np.asarray(gt)>127);crops[tid]=list(crop)
        photo_hi=np.asarray(photo.resize((hi.width,hi.height),Image.Resampling.LANCZOS))/255.
        gt_hi=np.asarray(gt.resize((hi.width,hi.height),Image.Resampling.NEAREST))>127
        rgb_image(contour(photo_hi,gt_hi)).save(panels/f'{tid}_GT.png',dpi=(300,300))
        for method in METHODS:
            ids_path=OUT/'selection_indices'/f'{label}_{method}.npy';ids=np.load(ids_path);ti=torch.as_tensor(ids,device='cuda')
            selected=scene.subset(ti)
            flags=torch.zeros_like(scene.xyz);flags[ti]=1
            with torch.no_grad():
                # Exact original-parameter contribution: audit only, not flattened
                # into the new RGB overlay, and never used to overwrite metrics.
                raw=c.raster(cam,scene,flags)[0].cpu().numpy()
                archived=np.asarray(Image.open(OUT/'projected_masks'/f'{tid}_{method}.png'))>127
                assert np.array_equal(raw>.05,archived),f'Projection mismatch: {tid}/{method}'
                np.save(maps/f'{tid}_{method}_original_contribution.npy',raw)
                Image.fromarray(np.uint16(np.clip(raw,0,1)*65535+.5)).save(maps/f'{tid}_{method}_original_contribution_16bit.png')
                shutil.copy2(OUT/'projected_masks'/f'{tid}_{method}.png',maps/f'{tid}_{method}_evaluation_mask_unchanged.png')
                # All selected splats retained. Same centers, anisotropic scales,
                # rotations and relative opacities. One scalar exposure uniformly
                # lowers only their display alpha; background depth occlusion is
                # still rendered with the original unselected Gaussians.
                logit=scene.opacity_logit.clone()
                logit[ti]=torch.logit((selected.opacity_logit.sigmoid()*EXPOSURE).clamp(1e-8,1-1e-8))
                display=c.GaussianData(scene.xyz,scene.log_scale,scene.rotation,logit,scene.sh)
                contribution=c.raster(hi,display,flags)[0].cpu().numpy()
                np.save(maps/f'{tid}_{method}_display_contribution.npy',contribution.astype(np.float16))
                Image.fromarray(np.uint16(np.clip(contribution,0,1)*65535+.5)).save(maps/f'{tid}_{method}_display_contribution_16bit.png')
                alpha=OVERLAY*contribution[...,None]
                overlay=photo_hi*(1-alpha)+np.array([1.,.10,.075])*alpha
                rgb_image(contour(overlay,gt_hi)).save(panels/f'{tid}_{method}_projection.png',dpi=(300,300))
                isolated=c.GaussianData(selected.xyz,selected.log_scale,selected.rotation,logit[ti],selected.sh)
                red=torch.ones_like(selected.xyz)*selected.xyz.new_tensor([1.,.19,.10])
                splats=c.raster(hi,isolated,red).permute(1,2,0).cpu().numpy()
                isolated_original_alpha=c.raster(hi,selected,torch.ones_like(selected.xyz))[0].cpu().numpy()
                original_rgb=c.raster(hi,selected,selected.colors(hi.center)).permute(1,2,0).cpu().numpy()
            rgb_image(splats).save(panels/f'{tid}_{method}_gaussian_only_full.png',dpi=(300,300))
            rgb_image(original_rgb).save(panels/f'{tid}_{method}_gaussian_only_original_rgb.png',dpi=(300,300))
            y,x=np.where(isolated_original_alpha>.005)
            if len(x):
                pad=int(.04*max(x.max()-x.min()+1,y.max()-y.min()+1))+4
                box=[max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(hi.width,int(x.max())+pad+1),min(hi.height,int(y.max())+pad+1)]
            else:box=[0,0,hi.width,hi.height]
            # Framing only: show the entire visible selection, including distant
            # false-positive components; never keep only the desired component.
            inset=ImageOps.pad(rgb_image(splats).crop(box),(720,405),method=Image.Resampling.LANCZOS,color=(0,0,0))
            inset.save(panels/f'{tid}_{method}_gaussian_only_inset.png',dpi=(300,300))
            support=contribution[contribution>1e-4]
            records.append(dict(task_id=tid,method=method,selected_gaussians=len(ids),indices_sha256=sha(ids_path),
                camera=cam.record(),display_camera=hi.record(),gaussian_only_inset_crop_pixels=box,
                original_evaluation_mask_identical=True,display_contribution_quantiles=np.quantile(support,[.1,.5,.9,.99]).tolist(),
                gaussian_geometry_modified=False,saved_opacity_modified=False,display_alpha_multiplier=EXPOSURE))
            print('Rendered actual splats:',tid,method,len(ids),'Gaussians',flush=True)
            del flags,selected,display,logit,isolated;torch.cuda.empty_cache()
    assert protected()==snapshot;assert sha(scene_path)==src['scene_sha256']
    write(OUT/'sources_revised.json',dict(original_sources=src,revision_scope='F17 panels a/b only; c and all scores are unchanged',
        original_files_sha256=snapshot,source_scene_sha256=src['scene_sha256'],source_scene_unchanged=True,
        common_crops=crops,render_records=records,render_resolution_scale=SCALE,
        main_projection='Continuous full-scene rasterized selected-Gaussian contribution; unselected scene maintains original occlusion.',
        display_only_opacity_exposure=EXPOSURE,overlay_max_alpha=OVERLAY,
        splat_geometry='All actual selected Gaussians and their original centers/scales/rotations retained; no sampling, jitter, invented holes, 2D blur or morphological processing of predictions.',
        inset='Selected-only real Gaussian rendering on black, same calibrated camera. Inset framing independently fits the full visible selection; no RGB photo or unselected Gaussians.',
        evaluation='Archived binary masks and IoU files preserved. Display exposure is never used for quantitative evaluation.',
        original_backup='Existing figure_F17.png/pdf, panels/, projected_masks/ and caption.txt retained unchanged.'))

def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    panels=OUT/'panels_revised';s=read(OUT/'sources_revised.json');d=read(OUT/'metrics.json')
    assert protected()==s['original_files_sha256']
    inset_crops={}
    for tid in QUAL:
        gt=np.asarray(Image.open(OUT/'ground_truth_masks'/tid/'mask.png'))>127
        yy,xx=np.where(gt);margin=max(8,round(.18*max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)))
        box=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),
            min(gt.shape[1],int(xx.max())+margin+1),min(gt.shape[0],int(yy.max())+margin+1)]
        inset_crops[tid]=box
        for method in METHODS:
            # Shared target-neighborhood crop for detail readability. Full-set
            # renders and original all-visible framing stay separately available.
            full=Image.open(panels/f'{tid}_{method}_gaussian_only_full.png')
            old_inset=panels/f'{tid}_{method}_gaussian_only_inset.png'
            all_visible=panels/f'{tid}_{method}_gaussian_only_all_visible_inset.png'
            if not all_visible.exists():shutil.copy2(old_inset,all_visible)
            detail=ImageOps.pad(full.crop(tuple(v*SCALE for v in box)),(720,405),
                method=Image.Resampling.LANCZOS,color=(0,0,0))
            detail.save(old_inset,dpi=(300,300))
    summary={r['method']:r for r in d['summary']};scores={(r['task_id'],r['method']):r for r in d['per_view']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.labelsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(180/25.4,133/25.4),facecolor='white')
    x=[.013,.260,.507,.754];width=.233;main_axes=[]
    for i,label in enumerate(['Ground Truth*']+METHODS):fig.text(x[i]+width/2,.982,label,ha='center',va='top',fontsize=8.5,fontweight='bold')
    for row,tid in enumerate(QUAL):
        crop=s['common_crops'][tid];crop_hi=tuple(int(v*SCALE) for v in crop);y=.689-row*.308
        fig.text(.014,y+.244,f'({chr(97+row)}) '+tid.split('_')[0]+('  |  Similar coolers' if row==0 else '  |  Partly occluded assembly stand'),fontsize=7.2,va='bottom')
        for col,method in enumerate(['GT']+METHODS):
            im=Image.open(panels/f'{tid}_{method if method=="GT" else method+"_projection"}.png')
            ax=fig.add_axes([x[col],y,width,.233]);main_axes.append((tid,method,ax));ax.imshow(im.crop(crop_hi));ax.axis('off')
            im.crop(crop_hi).save(panels/f'{tid}_{method}_main_crop.png',dpi=(300,300))
            inset=ax.inset_axes([.008,.008,.36,.27])
            if method=='GT':
                overview=Image.open(OUT/'panels'/f'{tid}_GT.png').convert('RGB');dr=ImageDraw.Draw(overview)
                dr.rectangle(tuple(crop),outline='white',width=2);inset.imshow(overview)
                label='Independent AI draft'
            else:
                inset.imshow(Image.open(panels/f'{tid}_{method}_gaussian_only_inset.png'))
                inset.text(.03,.97,'Gaussian-only',transform=inset.transAxes,ha='left',va='top',fontsize=4.3,color='white',bbox=dict(facecolor='black',alpha=.8,pad=.4,edgecolor='none'))
                label=f"IoU = {scores[(tid,method)]['primary_heldout_iou']:.3f}"
            inset.set_xticks([]);inset.set_yticks([])
            for spine in inset.spines.values():spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.7)
            fig.text(x[col]+width/2,y-.009,label,ha='center',va='top',fontsize=7)
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=4,label='Selected Gaussians (projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.4,label='GT contour (AI draft)')],loc='center',bbox_to_anchor=(.5,.329),ncol=2,frameon=False,fontsize=6.7)
    # Deliberately identical c layout, data and plotting calls to figure_f17.py.
    ax=fig.add_axes([.084,.074,.895,.20]);valid=d['common_targets'];xpos=np.arange(len(valid));bw=.21
    for j,(method,color) in enumerate(zip(METHODS,['#be7d34','#697b8f','#25799a'])):
        vals=[next(r['heldout_iou'] for r in d['per_target'] if r['label']==label and r['method']==method) for label in valid]
        ax.bar(xpos+(j-1)*bw,vals,bw*.93,color=color,label=f"{method} (mean {summary[method]['mean_heldout_iou']:.3f})",zorder=3)
    ax.set_xticks(xpos,valid);ax.set_ylim(0,1.12);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Held-out IoU');ax.set_xlabel('Physical equipment target')
    ax.grid(axis='y',color='#e9e9e9',zorder=0);ax.set_title('(c) Localization IoU on common eligible targets',loc='left',pad=6)
    ax.legend(loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.05),columnspacing=1.4,handlelength=1.3)
    fig.savefig(OUT/'figure_F17_revised.png',dpi=300);fig.savefig(OUT/'figure_F17_revised.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for tid,method,axis in main_axes:
        box=axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(panels/f'{tid}_{method}_with_inset.png',bbox_inches=box,dpi=300)
    plt.close(fig)
    old=np.asarray(Image.open(OUT/'figure_F17.png'));new=np.asarray(Image.open(OUT/'figure_F17_revised.png'))
    start=int(.70*old.shape[0]);assert np.array_equal(old[start:],new[start:]),'Panel c changed'
    shutil.copy2(OUT/'panels/c_statistics.png',panels/'c_statistics_unchanged.png')
    assert protected()==s['original_files_sha256']
    s.update(figure_mm=[180,133],dpi=300,vector_pdf_text=True,all_original_outputs_unchanged=True,
        c_pixel_identical_from_row=start,c_pixel_identical=True,main_crop_identical_across_methods=True,
        gaussian_only_detail_crops=inset_crops,
        inset='Same-camera selected-only Gaussian render on black; figure insets share one GT-neighborhood crop per case. Full visible selections, including off-crop false positives, are saved in gaussian_only_full.png and gaussian_only_all_visible_inset.png. No selection or geometry filtering.')
    write(OUT/'sources_revised.json',s)
    original_caption=(OUT/'caption.txt').read_text().strip()
    note=('Visualization revision for panels (a-b): red is a continuous anisotropic-Gaussian splat render, not a binary segmentation fill. '
        'A uniform display-only opacity multiplier of 0.12 and overlay gain 0.80 are applied equally to all three methods to expose density variation; '
        'the stored model, indices, covariance and all evaluation masks and scores are unchanged. '
        'The main overlay retains occlusion by the original unselected scene. Black-background Gaussian-only insets use the same calibrated camera, '
        'remove unselected scene Gaussians and use an identical local target-neighborhood crop for the three methods within each row. '
        'Full selected-set renders, including off-crop false positives, are also retained separately; the insets are local detail views rather than complete-selection coverage. '
        'The GT inset retains the complete photograph and marks the shared crop. '
        'No Gaussian subsampling, invented gaps, noise, 2D prediction blur or image repair is used. Panel (c) is unchanged. ')
    # Replace only the outdated visualization description in the separate caption.
    original_caption=original_caption.replace('The lower-left inset in every image preserves the complete photograph and all projected selections, including false positives outside the enlarged target region. ','')
    original_caption=original_caption.replace('Red indicates visible selected-Gaussian contribution;', 'Red indicates continuous selected-Gaussian display contribution;')
    (OUT/'caption_revised.txt').write_text(original_caption+'\n\n'+note+'\n',encoding='utf-8')
    (OUT/'code').mkdir(exist_ok=True);shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17_revised.pdf'),str(OUT/'pdf_proof_revised')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F17 - Gaussian projection revision</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:25px}img{max-width:100%}a{color:#155da1}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}</style><h1>F17 · Gaussian 投影可视化</h1><p>仅修改 a、b：连续 Gaussian 贡献叠加 + 黑底 Gaussian-only 小窗。所有方法使用相同的显示透明度倍率；模型、定位索引、IoU 数据与 c 图不变。</p><p><a href="figure_F17_revised.png">高清 PNG</a> · <a href="figure_F17_revised.pdf">PDF</a> · <a href="caption_revised.txt">图注</a> · <a href="sources_revised.json">参数及校验</a> · <a href="index.html">旧版检查页</a></p><img src="figure_F17_revised.png">'''
    for tid in QUAL:
        page+=f'<h2>{tid}</h2><div class="grid">'
        for method in METHODS:
            page+=f'<div><h3>{method}</h3><a href="panels_revised/{tid}_{method}_projection.png">完整 RGB + splats</a> · <a href="panels_revised/{tid}_{method}_gaussian_only_full.png">完整 Gaussian-only</a> · <a href="panels_revised/{tid}_{method}_gaussian_only_original_rgb.png">原参数/原色独立渲染</a><p>目标邻域 Gaussian-only：</p><img src="panels_revised/{tid}_{method}_gaussian_only_inset.png"><p>完整可见选区（含误选）：</p><img src="panels_revised/{tid}_{method}_gaussian_only_all_visible_inset.png"></div>'
        page+='</div>'
    page+='<p>'+html.escape(note)+'</p>'
    (OUT/'index_revised.html').write_text(page,encoding='utf-8')
    print('Revised figure written; old files and panel c verified unchanged.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['render','figure','all'],default='all');p.add_argument('--gpu',default='2');a=p.parse_args()
    if a.stage in ['render','all']:render(a.gpu)
    if a.stage in ['figure','all']:figure()
