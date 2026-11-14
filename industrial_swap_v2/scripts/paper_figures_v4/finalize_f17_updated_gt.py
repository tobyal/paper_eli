"""F17 final: two supplied GT updates, unchanged Gaussian predictions, mean IoU.

CPU-only evaluation and figure composition. Reuses actual cached splat renders;
never initializes a localization/generation model or Gaussian optimizer.
"""
import argparse, csv, hashlib, html, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
SRC=BASE/'F17_localization_comparison'
OUT=BASE/'F17_localization_comparison_final'
METHODS=['GaussianEditor','DGE','Ours']
QUAL=['M1-03_frame_000211','M1-05_frame_000228']
ATTACHMENTS={QUAL[0]:'codex-clipboard-4b0c3540-acce-454e-8b7e-4fe8664ba0ef.png',
             QUAL[1]:'codex-clipboard-389be52a-0001-45d7-973b-2bf3c94067c6.png'}

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def load_mask(p):
    im=Image.open(p);a=np.asarray(im)
    assert im.size==(938,528),f'Wrong original-image dimensions: {p}'
    if a.ndim==3:
        assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])
        if a.shape[2]==4:assert (a[:,:,3]==255).all()
        a=a[:,:,0]
    assert set(np.unique(a)).issubset({0,255}),f'Nonbinary mask: {p}'
    return a>127
def old_files():
    files=[p for p in SRC.iterdir() if p.suffix in ['.png','.pdf','.csv','.json','.txt']]
    for folder in ['ground_truth_masks','projected_masks','selection_indices']:
        files.extend(p for p in (SRC/folder).rglob('*') if p.is_file())
    for tid in QUAL:
        files.append(SRC/'photos'/f'{tid}.png')
        for m in METHODS:
            files.extend([SRC/'projected_masks_revised'/f'{tid}_{m}_display_contribution_16bit.png',
                SRC/'panels_revised'/f'{tid}_{m}_gaussian_only_full.png'])
    return {str(p.relative_to(SRC)):sha(p) for p in files}

def evaluate():
    for d in ['panels','ground_truth_masks','projected_masks','code']:(OUT/d).mkdir(parents=True,exist_ok=True)
    old=read(SRC/'metrics.json');views=read(SRC/'evaluation_views.json')['views'];vmap={v['task_id']:v for v in views}
    snapshot=old_files();rows=[];gt_records={};pred_records={}
    for task in QUAL:
        assert load_mask(OUT/'inputs'/f'{task}.png').any()
    # All available projections are rescored for traceability; the existing
    # primary inclusion set (6 views / 4 targets) is not changed by the GT update.
    for previous in old['per_view']:
        task=previous['task_id'];method=previous['method'];view=vmap[task]
        updated=task in QUAL
        src=OUT/'inputs'/f'{task}.png' if updated else SRC/'ground_truth_masks'/task/'mask.png'
        gt=load_mask(src);nonempty=bool(gt.any())
        if task not in gt_records:
            dest=OUT/'ground_truth_masks'/f'{task}.png';shutil.copy2(src,dest)
            gt_records[task]=dict(task_id=task,path=str(dest),source_path=str(src),sha256=sha(src),
                annotation_status='user_provided_updated_mask' if updated else 'unreviewed_AI_polygon_draft',
                original_attachment=ATTACHMENTS.get(task),updated=updated,nonempty=nonempty,
                primary_included=view['primary_included'],exclusion_reasons=view['exclusion_reasons'],
                camera=view['camera'],image_sha256=view['image_sha256'],original_GT_sha256=sha(SRC/'ground_truth_masks'/task/'mask.png'))
        pred_path=SRC/'projected_masks'/f'{task}_{method}.png'
        value=None;inter=None;union=None;pixels=None
        if nonempty and pred_path.exists():
            pred=load_mask(pred_path);inter=int((gt&pred).sum());union=int((gt|pred).sum());value=inter/union;pixels=int(pred.sum())
            # Independent implementation verifies the exported-mask arithmetic.
            aa=Image.open(pred_path).convert('1');bb=Image.open(src).convert('1')
            assert abs(value-sum(ImageChops.logical_and(aa,bb).getdata())/sum(ImageChops.logical_or(aa,bb).getdata()))<1e-12
            shutil.copy2(pred_path,OUT/'projected_masks'/pred_path.name)
            pred_records[f'{task}/{method}']=dict(path=str(pred_path),sha256=sha(pred_path),unchanged=True)
        eligible=bool(previous['primary_included']);assert not eligible or value is not None
        old_value=previous['supplied_draft_iou']
        if not updated and value is not None:assert abs(value-old_value)<1e-12
        rows.append(dict(task_id=task,label=previous['label'],m1_case_id=previous['m1_case_id'],
            physical_target_id=previous['physical_target_id'],method=method,camera=previous['camera'],
            primary_included=eligible,iou=value,primary_heldout_iou=value if eligible else None,
            previous_iou=old_value,iou_change=(value-old_value) if value is not None else None,
            annotation_status=gt_records[task]['annotation_status'],GT_updated=updated,
            GT_sha256=gt_records[task]['sha256'],GT_pixels=int(gt.sum()),prediction_pixels=pixels,
            intersection_pixels=inter,union_pixels=union,exclusion_reason=previous['exclusion_reason']))
    common=old['common_targets'];targets=[]
    for previous in old['per_target']:
        label=previous['label'];method=previous['method']
        rr=[r for r in rows if r['label']==label and r['method']==method and r['primary_included']]
        vals=[r['primary_heldout_iou'] for r in rr]
        targets.append(dict(physical_target_id=previous['physical_target_id'],label=label,
            m1_case_id=previous['m1_case_id'],description=previous['description'],method=method,
            heldout_iou=float(np.mean(vals)) if vals else None,valid_views=len(vals),
            updated_mask_views=sum(r['GT_updated'] for r in rr),
            remaining_AI_draft_views=sum(not r['GT_updated'] for r in rr),
            status='mixed_updated_and_AI_draft_annotations' if vals else 'NA_missing_eligible_GT'))
    assert [label for label in common if all(any(t['label']==label and t['method']==m and t['heldout_iou'] is not None for t in targets) for m in METHODS)]==common
    summary=[]
    for method in METHODS:
        vals=[t['heldout_iou'] for t in targets if t['method']==method and t['label'] in common]
        rr=[r for r in rows if r['primary_included'] and r['method']==method]
        assert len(vals)==4 and len(rr)==6 and sum(r['GT_updated'] for r in rr)==2
        summary.append(dict(method=method,mean_heldout_iou=float(np.mean(vals)),valid_equipment_count=4,
            valid_test_view_count=6,user_updated_mask_count=2,remaining_AI_draft_mask_count=4,
            previous_mean_heldout_iou=next(x['mean_heldout_iou'] for x in old['summary'] if x['method']==method),
            common_targets=';'.join(common),missing_equipment_excluded=5,
            aggregation='mean IoU across eligible views within each equipment target; then unweighted mean across four targets'))
    with (OUT/'iou_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    data=dict(per_view=rows,per_target=targets,summary=summary,common_targets=common)
    write(OUT/'metrics.json',data);write(OUT/'evaluation_views.json',dict(views=list(gt_records.values())))
    write(OUT/'sources.json',dict(previous_output_directory=str(SRC),source_revision=read(SRC/'sources_revised.json'),
        current_scope='Two user-supplied GT masks updated; same predictions and cohort; redesigned Gaussian visuals and target-macro mean chart.',
        annotation_update_date='2026-11-11',ground_truth=gt_records,prediction_sources=pred_records,
        old_files_sha256=snapshot,prediction_threshold=.05,evaluation_resolution=[938,528],
        mixed_annotation_status='Two user-provided updates and four unreviewed AI drafts among six primary views; not fully human-adjudicated.',
        heldout_scope='Unchanged: absent from per-target reference assistance and native tracing samples, not from scene/language-field training.',
        skipped_stages=['localization','generation','Gaussian optimization','new Gaussian rasterization'],
        reused_gaussian_visuals='Actual continuous splat contribution and selected-only rasterizations from the previous revision; no 2D prediction blur or synthetic texture.',
        metrics='metrics.json',mean_rule='within-target view mean, then unweighted mean across common valid physical equipment targets',
        unchanged_exclusions={v['task_id']:v['exclusion_reasons'] for v in views if not v['primary_included']},
        independent_Pillow_IoU_checks=len(pred_records),original_results_preserved=old_files()==snapshot))
    assert old_files()==snapshot
    for tid in QUAL:print(tid,[(m,next(r['iou'] for r in rows if r['task_id']==tid and r['method']==m)) for m in METHODS],flush=True)
    print('Updated means:',[(x['method'],x['mean_heldout_iou']) for x in summary],flush=True)

def main_crop(gt):
    yy,xx=np.where(gt);h,w=gt.shape
    cw=max(240,(xx.max()-xx.min()+1)*2.2,(yy.max()-yy.min()+1)*1.6*4/3);cw=min(cw,w,h*4/3);ch=cw*3/4
    x0=np.clip((xx.min()+xx.max())/2-cw/2,0,w-cw);y0=np.clip((yy.min()+yy.max())/2-ch/2,0,h-ch)
    return tuple(round(v) for v in [x0,y0,x0+cw,y0+ch])
def contour(rgb,gt,scale):
    out=rgb.copy();edge=binary_dilation(gt,iterations=scale)^binary_erosion(gt,iterations=scale)
    out[edge]=[0,210/255,230/255];return out
def image(a):return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))

def figure():
    d=read(OUT/'metrics.json');s=read(OUT/'sources.json');panels=OUT/'panels'
    assert old_files()==s['old_files_sha256']
    previous=read(SRC/'sources_revised.json');scale=previous['render_resolution_scale'];gain=previous['overlay_max_alpha']
    scores={(r['task_id'],r['method']):r for r in d['per_view']};summary={r['method']:r for r in d['summary']}
    crops={};inset_crops={};reused_assets={}
    # Recompose from the real continuous renderer output, never from a binary
    # prediction fill or an old RGB overlay containing the previous GT contour.
    for tid in QUAL:
        gt=load_mask(OUT/'ground_truth_masks'/f'{tid}.png');crops[tid]=list(main_crop(gt))
        photo=Image.open(SRC/'photos'/f'{tid}.png').convert('RGB');photo.save(panels/f'{tid}_original_photo.png')
        resolution=(photo.width*scale,photo.height*scale)
        raw=np.asarray(photo.resize(resolution,Image.Resampling.LANCZOS))/255.
        gth=np.asarray(Image.fromarray(np.uint8(gt)*255).resize(resolution,Image.Resampling.NEAREST))>127
        image(contour(raw,gth,scale)).save(panels/f'{tid}_GT.png',dpi=(300,300))
        yy,xx=np.where(gt);margin=max(10,round(.18*max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)))
        box=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(gt.shape[1],int(xx.max())+margin+1),min(gt.shape[0],int(yy.max())+margin+1)]
        inset_crops[tid]=box
        for method in METHODS:
            cp=SRC/'projected_masks_revised'/f'{tid}_{method}_display_contribution_16bit.png'
            contribution=np.asarray(Image.open(cp),dtype=np.float32)/65535.
            assert contribution.shape==gth.shape
            alpha=gain*contribution[...,None]
            rgb=raw*(1-alpha)+np.array([1.,.10,.075])*alpha
            image(contour(rgb,gth,scale)).save(panels/f'{tid}_{method}_projection.png',dpi=(300,300))
            fullpath=SRC/'panels_revised'/f'{tid}_{method}_gaussian_only_full.png'
            full=Image.open(fullpath).convert('RGB');shutil.copy2(fullpath,panels/fullpath.name)
            # Taller portrait inset makes real splat shape legible without
            # covering the centered target or stretching the image.
            crop=full.crop(tuple(int(v*scale) for v in box))
            inset=ImageOps.pad(crop,(540,720),method=Image.Resampling.LANCZOS,color='black')
            inset.save(panels/f'{tid}_{method}_gaussian_only_detail.png',dpi=(300,300))
            reused_assets[f'{tid}/{method}']=dict(contribution_path=str(cp),contribution_sha256=sha(cp),
                isolated_gaussians_path=str(fullpath),isolated_gaussians_sha256=sha(fullpath))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.labelsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(180/25.4,133/25.4),facecolor='white');panel_axes=[]
    x=[.013,.260,.507,.754];width=.233
    for i,label in enumerate(['2D GT']+METHODS):fig.text(x[i]+width/2,.982,label,ha='center',va='top',fontsize=8.5,fontweight='bold')
    for row,tid in enumerate(QUAL):
        crop=crops[tid];crop_hi=tuple(v*scale for v in crop);y=.689-row*.308
        fig.text(.014,y+.244,f'({chr(97+row)}) '+tid.split('_')[0]+('  |  Similar coolers' if row==0 else '  |  Partly occluded assembly stand'),fontsize=7.2,va='bottom')
        for col,method in enumerate(['GT']+METHODS):
            path=panels/f'{tid}_{method if method=="GT" else method+"_projection"}.png'
            im=Image.open(path);ax=fig.add_axes([x[col],y,width,.233]);panel_axes.append((tid,method,ax))
            ax.imshow(im.crop(crop_hi));ax.axis('off');im.crop(crop_hi).save(panels/f'{tid}_{method}_crop.png',dpi=(300,300))
            if method=='GT':
                small=Image.open(panels/f'{tid}_GT.png').resize((938,528),Image.Resampling.LANCZOS)
                ImageDraw.Draw(small).rectangle(tuple(crop),outline='white',width=2)
                ia=ax.inset_axes([.015,.015,.285,.215]);ia.imshow(small)
                subtitle='Updated GT mask'
            else:
                ia=ax.inset_axes([.015,.015,.285,.44]);ia.set_facecolor('black')
                ia.imshow(Image.open(panels/f'{tid}_{method}_gaussian_only_detail.png'),aspect='equal')
                ia.text(.5,1.025,'Gaussian-only',transform=ia.transAxes,ha='center',va='bottom',fontsize=4.4,color='white',
                    bbox=dict(facecolor='black',alpha=1,pad=1,edgecolor='none'),clip_on=False)
                subtitle=f"IoU = {scores[(tid,method)]['primary_heldout_iou']:.3f}"
            ia.set_xticks([]);ia.set_yticks([])
            for spine in ia.spines.values():spine.set_visible(True);spine.set_color('white');spine.set_linewidth(.7)
            fig.text(x[col]+width/2,y-.009,subtitle,ha='center',va='top',fontsize=7)
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=4,label='Selected Gaussians (projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.4,label='GT contour')],loc='center',bbox_to_anchor=(.5,.329),ncol=2,frameon=False,fontsize=6.7)
    ax=fig.add_axes([.084,.074,.895,.20])
    # Keep a clear center lane for the mean label; offsets are identical for all
    # methods and do not change the data values or the target order.
    offsets=np.array([-.21,-.13,.13,.21])
    assert len(d['common_targets'])==len(offsets)
    colors=['#be7d34','#697b8f','#25799a']
    for j,(method,color) in enumerate(zip(METHODS,colors)):
        val=summary[method]['mean_heldout_iou']
        vals=[next(t['heldout_iou'] for t in d['per_target'] if t['method']==method and t['label']==label) for label in d['common_targets']]
        ax.bar(j,val,.54,color=color,zorder=2)
        ax.scatter(j+offsets,vals,s=18,facecolor='white',edgecolor='#303030',linewidth=.6,zorder=4,clip_on=False)
        ax.text(j,val+.035,f'{val:.3f}',ha='center',va='bottom',fontsize=8,fontweight='bold',zorder=5)
    ax.set_xticks(range(3),METHODS);ax.set_xlim(-.6,2.6);ax.set_ylim(0,1.08);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylabel('IoU');ax.grid(axis='y',color='#e9e9e9',zorder=0)
    ax.set_title('(c) Mean Held-out Localization IoU',loc='left',pad=6)
    ax.text(.995,1.04,'4 devices / 6 test views',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.7,color='#444')
    ax.text(.995,.975,'Dots: per-device mean IoU',transform=ax.transAxes,ha='right',va='top',fontsize=6.2,color='#444')
    fig.savefig(OUT/'figure_F17_final.png',dpi=300);fig.savefig(OUT/'figure_F17_final.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for tid,method,axis in panel_axes:
        box=axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(panels/f'{tid}_{method}_with_inset.png',bbox_inches=box,dpi=300)
    box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.01,1.05)
    fig.savefig(panels/'c_mean_iou.png',bbox_inches=box,dpi=300);plt.close(fig)
    text=('Fig. 17. Gaussian-based industrial equipment localization with updated 2D annotations. '
        '(a-b) Actual Gaussian selections from GaussianEditor, DGE and Ours are projected onto the same registered photograph and common crop within each row. '
        'Cyan contours show the two user-provided updated GT masks. Red overlays are continuous Gaussian rasterization contributions, retaining the original centers, anisotropic covariance, selection membership and background occlusion. '
        'The common display-only opacity multiplier (0.12) and overlay gain (0.80) are unchanged from the previous visualization; no binary prediction fill, synthetic splat texture, random thinning or 2D prediction blur is applied. '
        'Black-background Gaussian-only insets show same-camera local detail with a shared target-neighborhood crop per row. Complete selected-set renders, including off-crop false positives, are also saved separately. '
        'IoU is measured over the full 938 x 528 test photograph using the original unmodified-parameter prediction masks at threshold 0.05, never using display-opacity maps. '
        '(c) Bars show unweighted mean IoU over four common valid physical equipment targets; each target first averages its eligible test views. Open circles show the four per-target means. '
        f"The target-macro means are {summary['GaussianEditor']['mean_heldout_iou']:.4f}, {summary['DGE']['mean_heldout_iou']:.4f} and {summary['Ours']['mean_heldout_iou']:.4f}, respectively. "
        'The cohort is unchanged: six eligible views of M1-03, M1-04, M1-05 and M1-06 in one industrial scene. Two views now use user-provided updated masks, while the other four retain unreviewed AI polygon drafts; the aggregate is therefore a mixed-annotation preliminary result, not a fully independently adjudicated benchmark. '
        'Five other physical targets lack eligible annotations and are excluded rather than scored as zero. The earlier exclusions are retained: M1-06/frame202 identifies the wrong physical instance, and M1-03/frame249 was used by native baseline tracing. '
        'Primary test views are held out from per-target photo assistance and sampled native tracing, but not from original scene/language-field training. '
        'GE and DGE use the same Text A, whereas Ours additionally uses the registered reference-photo instance cue. No localization or generation model is rerun, and all prior results are preserved.')
    (OUT/'caption.txt').write_text(text+'\n',encoding='utf-8')
    s.update(common_main_crops=crops,common_gaussian_only_detail_crops=inset_crops,reused_render_assets=reused_assets,
        figure_mm=[180,133],dpi=300,vector_PDF_text=True,visualization_opacity_exposure=previous['display_only_opacity_exposure'],
        visualization_overlay_gain=gain,primary_mean_device_count=4,primary_mean_view_count=6,original_results_preserved=old_files()==s['old_files_sha256'])
    assert s['original_results_preserved'];write(OUT/'sources.json',s)
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17_final.pdf'),str(OUT/'pdf_proof')],check=True)
    table=''.join(f"<tr><td>{m}</td><td>{summary[m]['mean_heldout_iou']:.4f}</td></tr>" for m in METHODS)
    page=f'''<!doctype html><meta charset="utf-8"><title>F17 - Updated GT and mean IoU</title><style>body{{font:16px/1.6 system-ui;max-width:1450px;margin:auto;padding:26px}}img{{max-width:100%}}a{{color:#155da1}}td,th{{padding:8px 24px;border-bottom:1px solid #ddd;text-align:right}}.note{{padding:14px;background:#f3f6f8}}</style>
<h1>F17 · 新 GT 与平均 IoU</h1><p class="note">已采用本次上传的两张 Mask，原始预测保持不变。主统计仍为 4 台设备 / 6 个合格视角，其中 2 张为本次用户更新标注，4 张保留 AI 草稿。缺失设备未计零，原排除规则不变。</p>
<p><a href="figure_F17_final.png">300 DPI PNG</a> · <a href="figure_F17_final.pdf">PDF</a> · <a href="iou_summary.csv">均值 CSV</a> · <a href="metrics.json">逐视角/逐设备数值</a> · <a href="caption.txt">英文图注</a> · <a href="sources.json">来源与校验</a> · <a href="../F17_localization_comparison/index_revised.html">上一版</a></p>
<img src="figure_F17_final.png"><table><tr><th>方法</th><th>平均 Held-out IoU</th></tr>{table}</table><p>{html.escape(text)}</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    (OUT/'README.txt').write_text('''F17 final: CPU-only reevaluation and figure composition.
Two user-supplied masks in inputs/ replace the GT for M1-03/frame211 and M1-05/frame228.
All other annotations, predictions, cameras and primary exclusions are preserved.
Main mean: four physical targets, six views (two updated masks + four AI drafts).
Average views within each target, then take an unweighted mean across targets.
No missing value is replaced by zero. Full-precision per-view and per-target scores are in metrics.json.
The displayed Gaussian opacity exposure is visualization-only and never used for IoU.
Full selected-only renders and local crops are both saved in panels/.
Original binary prediction masks are copied without modification into projected_masks/.

Reproduce on the server:
export ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli
cd "$ROOT"
envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/finalize_f17_updated_gt.py

The script reads the preceding F17 render cache. No GPU, localization, model generation
or Gaussian optimization is required. The old F17 directory remains unchanged.
''',encoding='utf-8')
    print('Final F17 figure exported; prior files unchanged.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','evaluate','figure'],default='all');a=p.parse_args()
    if a.stage in ['all','evaluate']:evaluate()
    if a.stage in ['all','figure']:figure()
