"""F17 presentation revision: soft-red splats, separate Gaussian row, mean bars.

Reuses the confirmed GT/evaluation files and selection indices. Only the
selected-only visualization is rasterized again; no localization or training.
The pre-existing final directory is archived intact before writing this version.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(os.environ.get('ROOT', str(Path(__file__).resolve().parents[3])))
BASE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4'
NATIVE = BASE / 'F17_localization_comparison'
OUT = BASE / 'F17_localization_comparison_final'
BACKUP = BASE / 'F17_localization_comparison_final_backup_20261111_two_rows'
TASK = 'M1-03_frame_000211'
METHODS = ['GaussianEditor', 'DGE', 'Ours']
COLOR = np.array([217, 120, 120], dtype=np.float32) / 255
SCALE, EXPOSURE, GAIN = 3, .12, .80
MAIN_CROP = [633, 184, 753, 334]
ONLY_CROP = [638, 194, 748, 326]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def image(array):
    return Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + .5))


def contour(rgb, gt):
    out = rgb.copy()
    edge = binary_dilation(gt, iterations=SCALE) ^ binary_erosion(gt, iterations=SCALE)
    out[edge] = [0, 210 / 255, 230 / 255]
    return out


def prepare():
    if not BACKUP.exists():
        assert OUT.is_dir() and (OUT / 'ground_truth_masks' / f'{TASK}.png').is_file()
        assert OUT.resolve().parent == BASE.resolve() == BACKUP.resolve().parent
        OUT.rename(BACKUP)
    for subdir in ['panels', 'renders', 'code']:
        (OUT / subdir).mkdir(parents=True, exist_ok=True)
    for name in ['metrics.json', 'iou_summary.csv', 'evaluation_views.json']:
        shutil.copy2(BACKUP / name, OUT / name)
    shutil.copy2(BACKUP / 'ground_truth_masks' / f'{TASK}.png', OUT / 'renders/updated_GT_mask.png')
    saved = {str(p.relative_to(BACKUP)): sha(p) for p in BACKUP.rglob('*') if p.is_file()}
    write(OUT / 'backup_manifest.json', {'directory': str(BACKUP), 'sha256': saved})


def render(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), ROOT=str(ROOT),
                      GSLE_OUTPUT=str(OUT / 'render_adapter'),
                      GSLE_RECORDS=str(OUT / 'render_records'), SWAP_DISABLE_QWEN='1')
    sys.path.insert(0, str(ROOT / 'gsle'))
    import torch
    import common as c

    native_sources = read(NATIVE / 'sources.json')
    scene_path = Path(native_sources['source_scene'])
    scene_hash = sha(scene_path)
    assert scene_hash == native_sources['scene_sha256']
    scene = c.GaussianData.load(scene_path)
    view = next(v for v in read(NATIVE / 'evaluation_views.json')['views'] if v['task_id'] == TASK)
    camera = c.Camera.from_record(view['camera'])
    hi = camera.resize(camera.width * SCALE, camera.height * SCALE)
    photo_path = NATIVE / 'photos' / f'{TASK}.png'
    photo = Image.open(photo_path).convert('RGB')
    raw = np.asarray(photo.resize((hi.width, hi.height), Image.Resampling.LANCZOS)) / 255.
    gt_path = OUT / 'renders/updated_GT_mask.png'
    gt = np.asarray(Image.open(gt_path).resize((hi.width, hi.height), Image.Resampling.NEAREST)) > 127
    image(contour(raw, gt)).save(OUT / 'renders/GT_full.png', dpi=(300, 300))
    render_records = []
    for method in METHODS:
        indices_path = NATIVE / 'selection_indices' / f'M1-03_{method}.npy'
        index_hash = sha(indices_path)
        indices = np.load(indices_path)
        selected = scene.subset(torch.as_tensor(indices, device='cuda'))
        cp = NATIVE / 'projected_masks_revised' / f'{TASK}_{method}_display_contribution_16bit.png'
        contribution = np.asarray(Image.open(cp), dtype=np.float32) / 65535.
        assert contribution.shape == gt.shape
        alpha = GAIN * contribution[..., None]
        image(contour(raw * (1 - alpha) + COLOR * alpha, gt)).save(
            OUT / 'renders' / f'{method}_scene_full.png', dpi=(300, 300))
        # Same selected splats, camera, covariance and uniform display exposure
        # as the accepted figure. The PLY/selection arrays are never modified.
        with torch.no_grad():
            logits = torch.logit((selected.opacity_logit.sigmoid() * EXPOSURE).clamp(1e-8, 1 - 1e-8))
            display = c.GaussianData(selected.xyz, selected.log_scale, selected.rotation, logits, selected.sh)
            color = torch.as_tensor(COLOR, device='cuda').expand_as(selected.xyz).contiguous()
            pixels = c.raster(hi, display, color).permute(1, 2, 0).cpu().numpy()
        image(pixels).save(OUT / 'renders' / f'{method}_gaussian_only_full.png', dpi=(300, 300))
        # Save an unquantized representation of this actual rasterization.
        np.save(OUT / 'renders' / f'{method}_gaussian_only_rgb.npy', pixels.astype(np.float16))
        assert sha(indices_path) == index_hash
        render_records.append(dict(method=method,selected_count=len(indices),indices=str(indices_path),
            indices_sha256=index_hash,contribution=str(cp),contribution_sha256=sha(cp),
            camera=camera.record(),display_camera=hi.record(),color_hex='#D97878',
            display_opacity_multiplier=EXPOSURE,scene_overlay_gain=GAIN,
            geometry_modified=False,selection_modified=False,stored_opacity_modified=False))
        print(f'{method}: rendered {len(indices)} unchanged selected Gaussians in #D97878', flush=True)
        del selected, display, logits, color
        torch.cuda.empty_cache()
    assert sha(scene_path) == scene_hash
    write(OUT / 'render_manifest.json', dict(source_scene=str(scene_path),scene_sha256=scene_hash,
        source_scene_unchanged=True,photo=str(photo_path),photo_sha256=sha(photo_path),
        updated_GT_sha256=sha(gt_path),records=render_records))


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    data = read(OUT / 'metrics.json')
    with (OUT / 'iou_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        summary = {r['method']: r for r in csv.DictReader(stream)}
    means = [float(summary[m]['mean_heldout_iou']) for m in METHODS]
    scores = {r['method']: r['primary_heldout_iou'] for r in data['per_view'] if r['task_id'] == TASK}
    assert all(int(summary[m]['valid_equipment_count']) == 4 and int(summary[m]['valid_test_view_count']) == 6 for m in METHODS)
    assert sha(OUT / 'iou_summary.csv') == sha(BACKUP / 'iou_summary.csv')
    panels = OUT / 'panels'
    for method in ['GT'] + METHODS:
        filename = 'GT_full.png' if method == 'GT' else f'{method}_scene_full.png'
        Image.open(OUT / 'renders' / filename).crop(tuple(v * SCALE for v in MAIN_CROP)).save(
            panels / f'{method}_scene.png', dpi=(300, 300))
        if method != 'GT':
            Image.open(OUT / 'renders' / f'{method}_gaussian_only_full.png').crop(
                tuple(v * SCALE for v in ONLY_CROP)).save(panels / f'{method}_gaussian_only.png', dpi=(300, 300))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    w, h = 180., 101.
    fig = plt.figure(figsize=(w / 25.4, h / 25.4), facecolor='white')
    def axis(x,y,aw,ah):
        return fig.add_axes([x / w, y / h, aw / w, ah / h])
    def text(x,y,s,**kwargs):
        return fig.text(x / w, y / h, s, **kwargs)
    text(1.5,98.7,'(a) M1-03 | Similar coolers',ha='left',va='center',fontsize=8.5)
    col_x = [1.5,35.,68.5,102.]
    pw = 31.5
    for i, method in enumerate(['GT'] + METHODS):
        text(col_x[i] + pw / 2,94,'2D GT' if method == 'GT' else method,
             ha='center',va='center',fontsize=7.8,fontweight='bold')
        ax = axis(col_x[i],51,pw,39.375)
        ax.imshow(Image.open(panels / f'{method}_scene.png'));ax.axis('off')
        label = 'Updated GT mask' if method == 'GT' else f'IoU = {scores[method]:.3f}'
        text(col_x[i] + pw / 2,48.5,label,ha='center',va='center',fontsize=7.1)
        if method != 'GT':
            text(col_x[i] + pw / 2,44,'Gaussian-only',ha='center',va='center',fontsize=6.8)
            ax = axis(col_x[i],3.5,pw,37.8)
            ax.imshow(Image.open(panels / f'{method}_gaussian_only.png'));ax.axis('off')
    # The otherwise empty GT cell carries only the shared two-color legend.
    la = axis(1.5,3.5,pw,37.8);la.axis('off')
    la.legend(handles=[Line2D([0],[0],color='#D97878',lw=3.5,label='Selected Gaussians\n(projection)'),
        Line2D([0],[0],color='#00d2e6',lw=1.3,label='GT contour')],loc='center',frameon=False,
        fontsize=6.0,handlelength=1.3,labelspacing=1.3,handletextpad=.6,borderpad=0)
    ax = axis(145,16,33.5,72)
    colors = ['#be7d34','#697b8f','#25799a']
    ax.bar(range(3),means,width=.55,color=colors,zorder=3)
    for i,v in enumerate(means):
        ax.text(i,v+.022,f'{v:.3f}',ha='center',va='bottom',fontsize=7.5,fontweight='bold')
    ax.set_ylim(0,1);ax.set_xlim(-.55,2.55)
    ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('IoU',fontsize=7,labelpad=1.5)
    ax.set_xticks(range(3),['Gaussian\nEditor','DGE','Ours'])
    ax.tick_params(axis='both',labelsize=6.3,length=2,pad=2)
    ax.grid(axis='y',color='#e9e9e9',zorder=0)
    text(161.75,97.1,'Mean Held-out\nLocalization IoU',ha='center',va='center',fontsize=7.7,linespacing=1.15)
    text(161.75,91,'4 devices / 6 test views',ha='center',va='center',fontsize=6,color='#444444')
    assert len(ax.patches) == 3 and len(ax.collections) == 0
    fig.savefig(OUT / 'figure_F17_final.png',dpi=300)
    fig.savefig(OUT / 'figure_F17_final.pdf')
    fig.canvas.draw()
    # Include the figure-level title and cohort note in the standalone chart.
    bbox = Bbox.from_extents(134.4 / 25.4, 9 / 25.4, 180 / 25.4, 101 / 25.4)
    fig.savefig(panels / 'mean_iou.png',bbox_inches=bbox,dpi=300)
    plt.close(fig)
    caption = ('Fig. 17. Gaussian-based industrial equipment localization. '
        'The M1-03 similar-cooler example compares the updated 2D ground truth, GaussianEditor, DGE and Ours. '
        'The upper row uses the same registered photograph, calibrated camera and shared crop. '
        'Cyan outlines the updated user-provided GT; semi-transparent soft red (#D97878) shows the continuous '
        'rasterized contribution of the actual selected Gaussians, with original background occlusion retained. '
        'The lower row shows selected-only Gaussian renders on black, using the same camera and identical local framing and scale across methods. '
        'Complete full-frame selected-set renders, including false positives outside this equipment-detail crop, are saved separately. '
        'All three methods use the same display-only opacity multiplier (0.12) and scene-overlay gain (0.80); '
        'no Gaussian geometry, stored opacity or selection is altered. Right: mean held-out IoU over four common '
        'valid equipment targets and six eligible test views. Views are averaged within each target, followed '
        'by an unweighted mean over targets; missing targets are not assigned zero. '
        'Two masks are user-provided updates and four remain unreviewed AI drafts, so the aggregate is a preliminary '
        'mixed-annotation result in one industrial scene. Evaluation uses the unchanged original full-frame masks '
        'at contribution threshold 0.05, independently of the display exposure. '
        'Only visualization color, selected-only rendering and figure layout are revised; no localization or generation experiment is rerun.')
    (OUT / 'caption.txt').write_text(caption + '\n',encoding='utf-8')
    render_sources = read(OUT / 'render_manifest.json')
    write(OUT / 'sources.json',dict(revision='Two image rows on left, mean-only chart on right; soft-red actual splat display',
        previous_final_backup=str(BACKUP),previous_compact_version=str(BASE / 'F17_localization_comparison_compact'),
        previous_sources=read(BACKUP / 'sources.json'),render_sources=render_sources,
        qualitative_case=TASK,main_crop=MAIN_CROP,gaussian_only_crop=ONLY_CROP,
        crops_in_original_image_pixels=True,render_resolution_scale=SCALE,figure_mm=[w,h],dpi=300,
        left_image_fraction=.75,gaussian_color_hex='#D97878',GT_color_hex='#00D2E6',
        same_main_crop_across_methods=True,same_gaussian_only_camera_crop_scale_across_methods=True,
        no_insets=True,mean_values=dict(zip(METHODS,means)),qualitative_IoU=scores,
        summary_CSV_byte_identical=True,bar_count=3,scatter_count=0,y_axis_limits=[0,1],
        annotation_caveat='2 user-updated and 4 unreviewed AI-draft masks; unchanged evaluation cohort',
        complete_selection_renders='renders/*_gaussian_only_full.png',
        lower_row_framing='Common local target detail; full visible selections retained separately without removal.'))
    manifest = read(OUT / 'backup_manifest.json')
    assert all(sha(BACKUP / path) == digest for path,digest in manifest['sha256'].items())
    write(OUT / 'validation.json',dict(old_final_backup_hashes_verified=True,metrics_unchanged=True,
        no_localization_or_training=True,selected_indices_and_scene_unchanged=True,
        actual_gaussian_rasterization=True,shared_camera=True,shared_crops=True,
        same_color_and_opacity_for_all_methods=True,no_insets=True,mean_bar_count=3,scatter_count=0,
        figure_size=list(Image.open(OUT / 'figure_F17_final.png').size)))
    shutil.copy2(__file__,OUT / 'code' / Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT / 'figure_F17_final.pdf'),str(OUT / 'pdf_proof')],check=True)
    page = '''<!doctype html><meta charset="utf-8"><title>F17 final - two rows</title>
<style>body{font:16px/1.6 system-ui;max-width:1600px;margin:auto;padding:24px}img{max-width:100%}a{color:#155da1}</style>
<h1>F17 · 两排高斯对比与均值图</h1><p>上排为同相机场景定位投影；下排为同相机、同尺度的独立高斯局部展示。统一柔和红色 #D97878。</p>
<p><a href="figure_F17_final.png">300 DPI PNG</a> · <a href="figure_F17_final.pdf">PDF</a> · <a href="iou_summary.csv">IoU CSV</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a> · <a href="../F17_localization_comparison_compact/index.html">上一紧凑版</a> · <a href="../F17_localization_comparison_final_backup_20261111_two_rows/index.html">原 final 备份</a></p>
<img src="figure_F17_final.png"><h2>独立子图</h2>'''
    for path in sorted(panels.glob('*.png')):
        page += f'<a href="panels/{path.name}">{path.stem}</a> &nbsp; '
    page += '<p>' + html.escape(caption) + '</p>'
    (OUT / 'index.html').write_text(page,encoding='utf-8')
    print('Final F17 exported; original final directory archived intact, CSV and experiment values unchanged.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','render','figure'],default='all')
    parser.add_argument('--gpu',default='2')
    args = parser.parse_args()
    if args.stage in ['all','render']:
        prepare();render(args.gpu)
    if args.stage in ['all','figure']:
        figure()
