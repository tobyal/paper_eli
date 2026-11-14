"""Update F17 from the user's red polygon; retain predictions and render weights.

No localization, model, optimization or synthetic image generation is run.
"""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT = Path(os.environ.get('ROOT', str(Path(__file__).resolve().parents[3])))
BASE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4'
OUT = BASE / 'F17_localization_comparison_final'
BACKUP = BASE / 'F17_localization_comparison_final_backup_20261111_soft_red'
NATIVE = BASE / 'F17_localization_comparison'
ATTACHMENT = ROOT / 'industrial_swap_v2/tmp/F17_GT_red_polygon.png'
TASK = 'M1-03_frame_000211'
METHODS = ['GaussianEditor', 'DGE', 'Ours']
SCALE, EXPOSURE, GAIN = 3, .12, .80
MAIN_CROP = [633,184,753,334]
ONLY_CROP = [638,194,748,326]
SCENE_RED = np.array([1., .10, .075])
ONLY_RED = np.array([1., .19, .10])
OLD_COLOR = np.array([217.,120.,120.]) / 255.


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def picture(a): return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))


def prepare():
    if not BACKUP.exists():
        assert (OUT/'renders/Ours_gaussian_only_rgb.npy').is_file()
        assert OUT.resolve().parent == BACKUP.resolve().parent == BASE.resolve()
        OUT.rename(BACKUP)
    for folder in ['inputs','ground_truth_masks','renders','panels','code']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ATTACHMENT,OUT/'inputs/user_red_polygon.png')
    write(OUT/'backup_manifest.json',dict(directory=str(BACKUP),
        sha256={str(p.relative_to(BACKUP)):sha(p) for p in BACKUP.rglob('*') if p.is_file()}))


def extract_gt():
    photo_path = NATIVE/'photos'/f'{TASK}.png'
    photo = Image.open(photo_path).convert('RGB')
    annotation = np.asarray(Image.open(ATTACHMENT).convert('RGB'))
    red = ((annotation[:,:,0]>210)&(annotation[:,:,1]<70)&(annotation[:,:,2]<70)).astype(np.uint8)
    # The one-pixel annotation has anti-aliased gaps at corners. Gather all
    # red stroke fragments within the visibly annotated equipment neighborhood
    # and recover its explicitly convex six-sided outline, not a rectangular ROI.
    annotation_roi = [1200,370,1340,580]
    roi = np.zeros_like(red)
    roi[annotation_roi[1]:annotation_roi[3],annotation_roi[0]:annotation_roi[2]] = 1
    yy,xx = np.where((red>0)&(roi>0))
    assert len(xx)>100
    boundary = cv2.convexHull(np.column_stack([xx,yy]).astype(np.int32))
    points = cv2.approxPolyDP(boundary,2,True).reshape(-1,2)
    assert len(points) == 6, f'Unexpected red outline: {points.tolist()}'
    assert .65 < xx.mean()/annotation.shape[1] < .8
    # Fill the actual external red polygon boundary; do not substitute its bbox.
    full = np.zeros(red.shape,np.uint8)
    cv2.drawContours(full,[boundary],-1,255,cv2.FILLED)
    gt = cv2.resize(full,photo.size,interpolation=cv2.INTER_NEAREST)>127
    assert gt.any() and photo.size == (938,528)
    reference = np.asarray(photo.resize((annotation.shape[1],annotation.shape[0]),Image.Resampling.LANCZOS))
    residual = np.abs(annotation.astype(float)-reference.astype(float)).mean(2)
    photo_mae = float(residual[red==0].mean())
    assert photo_mae < 10, f'Screenshot does not align with registered photograph: {photo_mae}'
    path = OUT/'ground_truth_masks'/f'{TASK}.png'
    Image.fromarray(np.uint8(gt)*255).save(path)
    shutil.copy2(path,OUT/'renders/updated_GT_mask.png')
    Image.fromarray(full).save(OUT/'inputs/polygon_mask_attachment_resolution.png')
    record = dict(annotation_source=str(ATTACHMENT),annotation_sha256=sha(ATTACHMENT),
        annotation_size=[annotation.shape[1],annotation.shape[0]],evaluation_size=list(photo.size),
        photo=str(photo_path),photo_sha256=sha(photo_path),alignment_mean_absolute_RGB_error=photo_mae,
        polygon_vertices_attachment=points.tolist(),polygon_vertices_evaluation=(points*np.array(photo.size)/np.array([annotation.shape[1],annotation.shape[0]])).tolist(),
        annotation_stroke_search_roi=annotation_roi,
        extraction='Convex hull of red stroke fragments in the marked target neighborhood; six-sided user polygon filled and nearest-neighbor resampled. ROI is only for stroke search, never used as GT. No semantic inference.',
        GT_sha256=sha(path),GT_pixels=int(gt.sum()),annotation_status='user_provided_red_polygon')
    write(OUT/'inputs/GT_extraction.json',record)
    print('GT polygon:',points.tolist(),'pixels:',int(gt.sum()),'photo alignment MAE:',photo_mae,flush=True)
    return gt,record


def evaluate(gt):
    data = read(BACKUP/'metrics.json')
    changes = []
    for row in data['per_view']:
        if row['task_id'] != TASK: continue
        path = NATIVE/'projected_masks'/f'{TASK}_{row["method"]}.png'
        pred = np.asarray(Image.open(path))>127
        assert pred.shape == gt.shape
        intersection = int((pred&gt).sum());union = int((pred|gt).sum())
        value = intersection/union
        before = row['iou']
        row.update(previous_revision_iou=before,iou=value,primary_heldout_iou=value if row['primary_included'] else None,
            GT_sha256=sha(OUT/'ground_truth_masks'/f'{TASK}.png'),GT_pixels=int(gt.sum()),
            intersection_pixels=intersection,union_pixels=union,annotation_status='user_provided_red_polygon',
            GT_updated=True,iou_change=value-row['previous_iou'])
        changes.append(dict(method=row['method'],before=before,after=value,prediction_sha256=sha(path)))
    for row in data['per_target']:
        values=[v['primary_heldout_iou'] for v in data['per_view'] if v['label']==row['label'] and v['method']==row['method'] and v['primary_included']]
        old=row['heldout_iou'];row['heldout_iou']=float(np.mean(values)) if values else None
        if row['label']!='M1-03': assert row['heldout_iou']==old
    for row in data['summary']:
        values=[v['heldout_iou'] for v in data['per_target'] if v['label'] in data['common_targets'] and v['method']==row['method']]
        assert len(values)==4 and all(v is not None for v in values)
        row['previous_revision_mean_heldout_iou']=row['mean_heldout_iou']
        row['mean_heldout_iou']=float(np.mean(values))
        assert row['valid_equipment_count']==4 and row['valid_test_view_count']==6
    write(OUT/'metrics.json',data)
    with (OUT/'iou_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data['summary'][0]))
        writer.writeheader();writer.writerows(data['summary'])
    views=read(BACKUP/'evaluation_views.json')
    for v in views['views']:
        if v['task_id']==TASK:
            v.update(path=str(OUT/'ground_truth_masks'/f'{TASK}.png'),source_path=str(OUT/'inputs/user_red_polygon.png'),
                sha256=sha(OUT/'ground_truth_masks'/f'{TASK}.png'),annotation_status='user_provided_red_polygon',
                updated=True,original_attachment=ATTACHMENT.name)
    write(OUT/'evaluation_views.json',views)
    write(OUT/'metric_changes.json',dict(changed_view=TASK,per_method=changes,
        means={v['method']:v['mean_heldout_iou'] for v in data['summary']},cohort_unchanged=True))
    print('Updated scores:',changes,flush=True)
    print('Updated means:',[(v['method'],v['mean_heldout_iou']) for v in data['summary']],flush=True)
    return data


def compose(gt):
    photo=Image.open(NATIVE/'photos'/f'{TASK}.png').convert('RGB')
    size=(photo.width*SCALE,photo.height*SCALE)
    rgb=np.asarray(photo.resize(size,Image.Resampling.LANCZOS))/255.
    hi_gt=np.asarray(Image.fromarray(np.uint8(gt)*255).resize(size,Image.Resampling.NEAREST))>127
    edge=binary_dilation(hi_gt,iterations=SCALE)^binary_erosion(hi_gt,iterations=SCALE)
    def draw(a):
        a=a.copy();a[edge]=[0,210/255,230/255];return picture(a)
    draw(rgb).save(OUT/'renders/GT_full.png',dpi=(300,300))
    records=[]
    for method in METHODS:
        cp=NATIVE/'projected_masks_revised'/f'{TASK}_{method}_display_contribution_16bit.png'
        contribution=np.asarray(Image.open(cp),dtype=np.float32)/65535.
        alpha=GAIN*contribution[...,None]
        draw(rgb*(1-alpha)+SCENE_RED*alpha).save(OUT/'renders'/f'{method}_scene_full.png',dpi=(300,300))
        cached=BACKUP/'renders'/f'{method}_gaussian_only_rgb.npy'
        pixels=np.load(cached).astype(np.float32)
        # For constant-color splats over black, RGB = color * accumulated alpha.
        # Recover the scalar from all three unquantized channels (float16 cache).
        # This is linear recoloring of actual rasterization, not mask synthesis.
        weights=(pixels*OLD_COLOR).sum(2)/(OLD_COLOR**2).sum()
        err=float(np.abs(pixels-weights[...,None]*OLD_COLOR).max())
        assert err<.001
        weights=np.clip(weights,0,1)
        picture(weights[...,None]*ONLY_RED).save(OUT/'renders'/f'{method}_gaussian_only_full.png',dpi=(300,300))
        np.save(OUT/'renders'/f'{method}_gaussian_only_alpha.npy',weights.astype(np.float16))
        records.append(dict(method=method,cached_actual_render=str(cached),cached_render_sha256=sha(cached),
            linear_color_residual_max=err,contribution_path=str(cp),contribution_sha256=sha(cp)))
    for method in ['GT']+METHODS:
        name='GT_full.png' if method=='GT' else f'{method}_scene_full.png'
        Image.open(OUT/'renders'/name).crop(tuple(v*SCALE for v in MAIN_CROP)).save(OUT/'panels'/f'{method}_scene.png',dpi=(300,300))
        if method!='GT':
            Image.open(OUT/'renders'/f'{method}_gaussian_only_full.png').crop(tuple(v*SCALE for v in ONLY_CROP)).save(OUT/'panels'/f'{method}_gaussian_only.png',dpi=(300,300))
    return records


def figure(data,gt_source,render_records):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    w,h=180.,101.
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    def axis(x,y,aw,ah):return fig.add_axes([x/w,y/h,aw/w,ah/h])
    def text(x,y,s,**kw):return fig.text(x/w,y/h,s,**kw)
    scores={r['method']:r['primary_heldout_iou'] for r in data['per_view'] if r['task_id']==TASK}
    means={r['method']:r['mean_heldout_iou'] for r in data['summary']}
    text(1.5,98.7,'(a) M1-03 | Similar coolers',ha='left',va='center',fontsize=8.5)
    cols=[1.5,35.,68.5,102.];pw=31.5
    for i,method in enumerate(['GT']+METHODS):
        text(cols[i]+pw/2,94,'2D GT' if method=='GT' else method,ha='center',va='center',fontsize=7.8,fontweight='bold')
        ax=axis(cols[i],51,pw,39.375);ax.imshow(Image.open(OUT/'panels'/f'{method}_scene.png'));ax.axis('off')
        text(cols[i]+pw/2,48.5,'Updated GT polygon' if method=='GT' else f'IoU = {scores[method]:.3f}',ha='center',va='center',fontsize=7.1)
        if method!='GT':
            text(cols[i]+pw/2,44,'Gaussian-only',ha='center',va='center',fontsize=6.8)
            ax=axis(cols[i],3.5,pw,37.8);ax.imshow(Image.open(OUT/'panels'/f'{method}_gaussian_only.png'));ax.axis('off')
    ax=axis(1.5,3.5,pw,37.8);ax.axis('off')
    ax.legend(handles=[Line2D([0],[0],color='#e34c46',lw=3.5,label='Selected Gaussians\n(projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.3,label='GT contour')],loc='center',frameon=False,
        fontsize=6,handlelength=1.3,labelspacing=1.3,handletextpad=.6,borderpad=0)
    # Independent 44 x 33 mm chart block: exact overall 4:3 aspect ratio.
    cx,cy,cw,ch=136.,34.,44.,33.
    ax=axis(cx+5.5,cy+5.5,37.,17.)
    values=[means[m] for m in METHODS]
    ax.bar(range(3),values,.55,color=['#be7d34','#697b8f','#25799a'],zorder=3)
    for i,v in enumerate(values):ax.text(i,v+.035,f'{v:.3f}',ha='center',va='bottom',fontsize=6.5,fontweight='bold')
    ax.set_xlim(-.55,2.55);ax.set_ylim(0,1);ax.set_yticks([0,.5,1]);ax.set_ylabel('IoU',fontsize=6,labelpad=1)
    ax.set_xticks(range(3),['Gaussian\nEditor','DGE','Ours']);ax.tick_params(axis='both',labelsize=5.7,length=2,pad=2)
    ax.grid(axis='y',color='#e9e9e9',zorder=0)
    text(cx+cw/2,cy+30,'Mean Held-out\nLocalization IoU',ha='center',va='center',fontsize=7,linespacing=1.1)
    text(cx+cw/2,cy+25.4,'4 devices / 6 test views',ha='center',va='center',fontsize=5.8,color='#444444')
    assert len(ax.patches)==3 and len(ax.collections)==0
    fig.savefig(OUT/'figure_F17_final.png',dpi=300);fig.savefig(OUT/'figure_F17_final.pdf')
    # Include the y-axis label within the exported 4:3 chart, not just its axes.
    chart_bounds=[134.,33.25,180.,67.75]
    fig.savefig(OUT/'panels/mean_iou.png',bbox_inches=Bbox.from_extents(*(v/25.4 for v in chart_bounds)),dpi=600)
    plt.close(fig)
    caption=('Fig. 17. Gaussian-based industrial equipment localization. The M1-03 similar-cooler example '
        'compares the user-updated 2D GT polygon with GaussianEditor, DGE and Ours. The upper row uses '
        'the same registered photograph, camera and crop. Cyan shows the polygon enclosed by the red '
        'outline in the user-provided annotation, mapped to the original evaluation resolution; red '
        'shows actual selected-Gaussian contribution, retaining scene occlusion. The lower row shows '
        'selected-only Gaussian detail on black with identical camera, framing, scale and display opacity '
        'across methods. Full-frame selected sets, including off-crop false positives, are saved separately. '
        'The former red display colors are restored; the opacity multiplier (0.12) and scene-overlay gain '
        '(0.80) are unchanged. Right: mean held-out localization IoU in a 4:3 chart block. Scores are '
        'recomputed for the updated M1-03 GT using the same stored full-frame prediction masks at threshold '
        '0.05. The unchanged cohort comprises four common physical targets and six eligible views; '
        'view scores are first averaged within target, then averaged equally across targets. Two masks '
        'are user annotations and four remain unreviewed AI drafts; results remain preliminary mixed-annotation '
        'evaluation in one scene. No localization, generation, geometry, selection or optimization is rerun.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    previous=read(BACKUP/'sources.json')
    write(OUT/'sources.json',dict(revision='User-red-polygon GT, restored prior red colors, 4:3 right chart',
        previous_final_backup=str(BACKUP),previous_sources=previous,updated_GT=gt_source,
        previous_render_manifest=read(BACKUP/'render_manifest.json'),color_recomposition=render_records,
        scene_red_RGB=SCENE_RED.tolist(),gaussian_only_red_RGB=ONLY_RED.tolist(),
        same_settings_across_methods=True,display_opacity_multiplier=EXPOSURE,overlay_gain=GAIN,
        main_crop=MAIN_CROP,gaussian_only_crop=ONLY_CROP,camera_unchanged=True,
        figure_mm=[w,h],chart_mm=[46.,34.5],chart_overall_aspect_ratio='4:3',chart_position_mm=chart_bounds[:2],
        chart_uses_only_mean_bars=True,dpi=300,mean_IoU=means,qualitative_IoU=scores,
        valid_targets=4,valid_views=6,predictions_unchanged=True))
    saved=read(OUT/'backup_manifest.json')['sha256']
    assert all(sha(BACKUP/p)==v for p,v in saved.items())
    write(OUT/'validation.json',dict(previous_version_preserved=True,old_files_hashes_verified=True,
        user_polygon_six_vertices=True,photo_alignment_verified=True,predictions_unchanged=True,
        changed_evaluation_view=TASK,unchanged_other_target_scores=True,cohort_unchanged=True,
        chart_aspect_ratio=cw/ch,mean_bar_count=3,scatter_count=0,
        real_rasterized_alpha_reused=True,no_models_or_training=True,
        figure_pixels=list(Image.open(OUT/'figure_F17_final.png').size)))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F17_final.pdf'),str(OUT/'pdf_proof')],check=True)
    page='''<!doctype html><meta charset="utf-8"><title>F17 - updated polygon GT</title>
<style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F17 · 红线 GT 更新与 4:3 统计图</h1><p>使用用户红色多边形作为新 GT；高斯恢复之前的红色；右侧统计图为独立 4:3 图块。</p>
<p><a href="figure_F17_final.png">300 DPI PNG</a> · <a href="figure_F17_final.pdf">PDF</a> · <a href="iou_summary.csv">更新 IoU</a> · <a href="ground_truth_masks/M1-03_frame_000211.png">新 GT mask</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F17_localization_comparison_final_backup_20261111_soft_red/index.html">上一版备份</a></p>
<img src="figure_F17_final.png">'''
    page+='<p>'+html.escape(caption)+'</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('Updated F17 exported; old files and predictions preserved.',flush=True)


if __name__=='__main__':
    prepare()
    gt,gt_source=extract_gt()
    data=evaluate(gt)
    records=compose(gt)
    figure(data,gt_source,records)
