"""Reorganize saved Case00 F11 into three conditions and one analysis strip.

All annotations are projections of saved geometry/masks. CPU visualization only.
The approved historical Qwen result is explicitly distinguished from the later
adaptive conditions. Original input files, previous figures and metrics survive.
"""
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4'
PREVIOUS = BASE / 'F11_spatial_conditioning_case00'
SOURCE = PREVIOUS / 'inputs'
OUT = BASE / 'F11_spatial_conditioning_revised'
TOP_CROP = [100, 270, 940, 900]
ANALYSIS_CROP = [170, 240, 930, 810]
GUARD_CROP = [300, 690, 500, 890]
MAGENTA, ORANGE, CYAN, GREEN = '#D42BCE', '#E9A32F', '#00BEC9', '#55A777'
EDGES = [(i, i ^ b) for i in range(8) for b in (1, 2, 4) if i < (i ^ b)]


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def project(points, cam):
    xyz = np.asarray(points) @ np.asarray(cam['R']).T + cam['t']
    assert np.all(xyz[:, 2] > 0)
    return xyz[:, :2] / xyz[:, 2:] * [cam['fx'], cam['fy']] + [cam['cx'], cam['cy']]


def main():
    OUT.mkdir(exist_ok=True)
    for name in ['inputs', 'panels', 'analysis', 'code']:
        (OUT/name).mkdir(exist_ok=True)
    hashes = {str(p.relative_to(PREVIOUS)): sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    for p in SOURCE.iterdir():
        if p.is_file():
            shutil.copy2(p, OUT/'inputs'/p.name)
            assert sha(p) == sha(OUT/'inputs'/p.name)
    sources_before = read(PREVIOUS/'sources.json')
    condition = read(SOURCE/'conditions.json')
    region = read(SOURCE/'generation_region.json')
    camera = condition['camera']
    assert read(SOURCE/'anchor_camera.json') == camera
    assert read(SOURCE/'confirmed_generation_conditions.json')['camera'] == camera
    accepted = read(ROOT/'experiment figure/F05/sources.json')['qwen']
    assert sha(SOURCE/'edited_scene.png') == accepted['sha256']
    assert sha(SOURCE/'edit_mask.png') != sha(SOURCE/'confirmed_generation_edit_mask.png')

    rgb = np.asarray(Image.open(SOURCE/'anchor.png').convert('RGB'))
    generated = np.asarray(Image.open(SOURCE/'edited_scene.png').convert('RGB'))
    hint = np.asarray(Image.open(SOURCE/'position_hint.png').convert('RGB'))
    depth = np.asarray(Image.open(SOURCE/'depth_background.png').convert('RGB'))
    original_depth = np.asarray(Image.open(SOURCE/'depth.png').convert('RGB'))
    assert rgb.shape == generated.shape == hint.shape == depth.shape == (1024, 1024, 3)
    masks = {n: np.asarray(Image.open(SOURCE/(n+'.png'))) > 0 for n in [
        'edit_mask', 'protected_foreground_mask', 'extension_proposal', 'generation_bbox_mask']}
    edit = masks['edit_mask']
    guard = masks['protected_foreground_mask'] & masks['extension_proposal']
    assert not (edit & guard).any()
    assert not depth[edit].any() and np.array_equal(depth[~edit], original_depth[~edit])
    assert int(guard.sum()) == read(SOURCE/'mask_report.json')['protected_foreground_pixels']

    # Both extents come from generation_region.json. The OLD extent is the
    # stored measurement of selected M1 Gaussians in its oriented local frame,
    # not a manually adjusted box or the legacy world-axis-aligned M1 box.
    geometry = region['geometry']
    old_local = np.asarray(list(itertools.product(*zip(geometry['old_local_low'], geometry['old_local_high']))))
    new_local = np.asarray(list(itertools.product(*zip(geometry['new_local_low'], geometry['new_local_high']))))
    axes = np.asarray(geometry['axes'])
    origin = np.asarray(geometry['origin'])
    old_world = old_local @ axes.T + origin
    reconstructed_new = new_local @ axes.T + origin
    new_world = np.asarray(region['bbox_corners'])
    pair_dist = np.linalg.norm(reconstructed_new[:, None] - new_world[None], axis=2)
    assert np.max(pair_dist.min(axis=1)) < 1e-7
    old_xy, new_xy = project(old_world, camera), project(new_world, camera)
    bbox_mask = np.zeros((1024, 1024), np.uint8)
    cv2.fillConvexPoly(bbox_mask, cv2.convexHull(np.round(new_xy).astype('int32')), 1)
    assert np.array_equal(bbox_mask.astype(bool), masks['generation_bbox_mask'])
    local_in_frame = (new_world-region['world_transform']['center']) @ np.asarray(region['world_transform']['axes'])
    bottom_ids = np.flatnonzero(np.isclose(local_in_frame[:, 1], local_in_frame[:, 1].min()))
    support = new_xy[bottom_ids]
    center = support.mean(0)
    support = support[np.argsort(np.arctan2(support[:, 1]-center[1], support[:, 0]-center[0]))]

    shown_hint = hint.copy()
    shown_hint[guard] = (.4*hint[guard] + .6*np.array([0, 222, 231])).astype('uint8')
    Image.fromarray(shown_hint).save(OUT/'analysis/position_hint_protection_overlay.png')
    # Display contrast only; the supplied depth file remains byte-for-byte intact.
    assert np.array_equal(depth[:, :, 0], depth[:, :, 1]) and np.array_equal(depth[:, :, 0], depth[:, :, 2])
    gray = depth[:, :, 0]
    valid = (~edit) & (gray > 0)
    lo, hi = np.percentile(gray[valid], [2, 98])
    depth_display = np.zeros_like(gray)
    depth_display[valid] = np.rint(8 + 247*np.clip((gray[valid].astype(float)-lo)/(hi-lo), 0, 1)).astype('uint8')
    assert not depth_display[edit].any()
    Image.fromarray(depth_display).save(OUT/'analysis/depth_display_contrast.png')
    contrast = dict(display_only=True, mapping='8 + 247 * clip((value - p2)/(p98 - p2),0,1), rounded to uint8',
                    percentiles=[2,98], lower_gray=float(lo), upper_gray=float(hi),
                    edit_pixels_black=True, original_zero_pixels_preserved=True, original_depth_file_modified=False)
    difference = np.abs(generated.astype(np.float32)-rgb.astype(np.float32)).mean(2)/255.
    np.save(OUT/'analysis/rgb_difference.npy', difference)
    previous_metrics = read(PREVIOUS/'analysis/metrics.json')
    inside, outside = float(difference[edit].mean()), float(difference[~edit].mean())
    assert abs(inside-previous_metrics['inside_edit_rgb_mae']) < 1e-9
    assert abs(outside-previous_metrics['outside_edit_rgb_mae']) < 1e-9
    write(OUT/'analysis/metrics.json', previous_metrics)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42})
    W, H = 180., 122.
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
    xs, panel_w, image_h = [2., 61.5, 121.], 57., 42.75

    def axis(x, y, w, h):
        return fig.add_axes([x/W, y/H, w/W, h/H])

    def text(x, y, value, **kw):
        return fig.text(x/W, y/H, value, **kw)

    def display(ax, arr, crop, **kw):
        ax.imshow(arr, extent=[-.5, 1023.5, 1023.5, -.5], interpolation='lanczos', **kw)
        ax.set_xlim(crop[0]-.5, crop[2]-.5)
        ax.set_ylim(crop[3]-.5, crop[1]-.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def wire(ax, pts, color, lw, style='solid'):
        for i, j in EDGES:
            ax.plot(pts[[i,j],0], pts[[i,j],1], color=color, lw=lw, linestyle=style,
                    solid_capstyle='round', dash_capstyle='round')

    titles = ['(a) 3D Workspace', '(b) Visibility-Aware Edit Region', '(c) Background Depth']
    for x, title in zip(xs, titles):
        text(x, 116.3, title, fontsize=7.6, fontweight='semibold', ha='left', va='center')
    a = axis(xs[0], 69, panel_w, image_h)
    display(a, rgb, TOP_CROP)
    a.add_patch(Polygon(support, closed=True, facecolor=GREEN, edgecolor=GREEN, linewidth=.8, alpha=.29))
    wire(a, old_xy, ORANGE, 1.0, (0, (3, 2)))
    wire(a, new_xy, MAGENTA, 1.12)
    fig.legend(handles=[Line2D([0],[0], color=ORANGE, lw=1.2, linestyle='--', label='Old'),
                        Line2D([0],[0], color=MAGENTA, lw=1.2, label='New'),
                        Patch(facecolor=GREEN, alpha=.5, label='Support')],
               loc='center', bbox_to_anchor=((xs[0]+panel_w/2)/W,65.8/H), ncol=3,
               frameon=False, fontsize=6.2, handlelength=1.5, handletextpad=.5, columnspacing=1.1)

    b = axis(xs[1], 69, panel_w, image_h)
    display(b, shown_hint, TOP_CROP)
    b.contour(guard, levels=[.5], colors=[CYAN], linewidths=.55)
    gx,gy,gxx,gyy = GUARD_CROP
    b.add_patch(Rectangle((gx,gy),gxx-gx,gyy-gy,fill=False,edgecolor='white',linewidth=.8,linestyle=(0,(3,2))))
    # Top-right detail window avoids the projected target and generated volume.
    inset_size, header_h = 18.5, 2.9
    inset_x, inset_y = xs[1]+panel_w-inset_size-1., 69+image_h-inset_size-header_h-1.
    ia = axis(inset_x,inset_y,inset_size,inset_size)
    display(ia,shown_hint,GUARD_CROP)
    ia.contour(guard,levels=[.5],colors=[CYAN],linewidths=.65)
    for spine in ia.spines.values():
        spine.set_visible(True);spine.set_color('#eeeeee');spine.set_linewidth(.65)
    ia.add_patch(Rectangle((0,1),1,header_h/inset_size,transform=ia.transAxes,clip_on=False,
                           facecolor='white',edgecolor='#eeeeee',linewidth=.65))
    ia.text(.5,1+header_h/(2*inset_size),'Foreground detail',ha='center',va='center',
            transform=ia.transAxes,fontsize=5.5,color='black',clip_on=False)
    fig.legend(handles=[Line2D([0],[0],color=MAGENTA,lw=3,label='Edit region'),
                        Line2D([0],[0],color=CYAN,lw=3,label='Protected foreground')],
               loc='center',bbox_to_anchor=((xs[1]+panel_w/2)/W,65.8/H),ncol=2,frameon=False,
               fontsize=6.,handlelength=1.3,handletextpad=.4,columnspacing=.9)
    c=axis(xs[2],69,panel_w,image_h)
    display(c,depth_display,TOP_CROP,cmap='gray',vmin=0,vmax=255)
    text(xs[2]+panel_w/2,65.8,'Black: Unknown Depth',ha='center',va='center',fontsize=6.5)

    text(2,59.2,'(d) Localized Editing Analysis',ha='left',va='center',fontsize=8.,fontweight='semibold')
    analysis_axes=[]
    for x,label,data in zip(xs,['Original','Generated*','|ΔRGB|'],[rgb,generated,difference]):
        text(x+panel_w/2,54.7,label,ha='center',va='center',fontsize=7.4)
        da=axis(x,9.,panel_w,image_h)
        if data.ndim==2:
            display(da,data,ANALYSIS_CROP,cmap='inferno',vmin=0,vmax=.5)
        else:
            display(da,data,ANALYSIS_CROP)
        analysis_axes.append(da)
    text(2,4.3,f'Inside-edit RGB Difference: {inside:.3f}     Outside-edit RGB Difference: {outside:.3f}',
         ha='left',va='center',fontsize=6.3)
    cbax=axis(126,4.3,47,1.3)
    cb=fig.colorbar(analysis_axes[2].images[0],cax=cbax,orientation='horizontal',ticks=[0,.25,.5])
    cb.ax.set_xticklabels(['0','0.25','0.5+']);cb.ax.tick_params(labelsize=5.5,length=1.3,pad=1.)
    cb.outline.set_visible(False)

    fig.savefig(OUT/'figure_F11_revised.png',dpi=300)
    fig.savefig(OUT/'figure_F11_revised.pdf')
    bounds={'a_3d_workspace':[1,63.5,60,119.7],
            'b_visibility_aware_edit_region':[60.5,63.5,119.5,119.7],
            'c_background_depth':[120,63.5,179,119.7],
            'd_localized_editing_analysis':[1,.2,179,61.8]}
    for name,box in bounds.items():
        fig.savefig(OUT/'panels'/(name+'.png'),dpi=600,
                    bbox_inches=Bbox.from_extents(*[v/25.4 for v in box]))
    for x,name in zip(xs,['d_original','d_generated','d_rgb_difference']):
        box=[x,8.8,x+panel_w,56.4]
        fig.savefig(OUT/'panels'/(name+'.png'),dpi=600,
                    bbox_inches=Bbox.from_extents(*[v/25.4 for v in box]))
    plt.close(fig)

    caption=(
        'Fig. 11. Spatial and visibility-aware conditioning for localized industrial equipment generation, illustrated with Case00. '
        '(a) The measured old equipment extent (orange dashed), new generation volume (magenta) and lower support face '
        '(translucent green) are projected into the original 3DGS Anchor render. All three derive from the saved '
        'generation_region.json: the old extent is the stored geometry.old_local_low/high in the measured oriented frame, '
        'the new volume is bbox_corners and the support is its local-y bottom face. This is a geometric overlay; no floor '
        'Gaussians or geometry were created. (b) Saved magenta position hint with the local protected foreground in cyan. '
        'The enlargement shows the foreground roller-rack corner; its source crop is indicated by the dashed box. '
        'Cyan and the inset are explanatory figure annotations, not extra generation inputs. No mask boundary was changed. '
        '(c) The current background-depth reference, with a display-only 2nd-to-98th percentile contrast stretch of positive '
        'non-edit pixels. Black edit pixels remain unknown depth; source depth data are unmodified. '
        '(d) Identically cropped original and confirmed generated RGB, followed by mean per-channel absolute RGB difference '
        'in [0,1]. The heatmap saturates at 0.5. Reported inside/outside values (0.216 and 0.018) retain the previous full-image '
        'partition by the current edit mask, rather than being recomputed on the display crop. '
        '*The generated robot is the previously approved main-pipeline reference used in F05, produced with earlier '
        'projected-box conditions. It predates the adaptive conditions in (a-c) and is not their direct generated output. '
        'The same saved Anchor camera and 1024-by-1024 canvas are shared by the current conditions and both RGB images; '
        'no registration, recoloring or image repair was applied. This figure visualizes existing conditions and the '
        'confirmed historical reference, rather than demonstrating a new conditioning ablation or causal preservation '
        'effect. No M1-M4 stage, generation model, or new Gaussian rendering was run.')
    (OUT/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    write(OUT/'sources.json',dict(case='case00',previous_version=str(PREVIOUS),
        previous_sources=sources_before,source_directory=str(SOURCE),
        source_hashes={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},
        confirmed_generation_source=accepted,updated_conditions_used_for_confirmed_generation=False,
        camera=camera,source_resolution=[1024,1024],top_common_crop=TOP_CROP,
        analysis_common_crop=ANALYSIS_CROP,foreground_detail_crop=GUARD_CROP,
        geometry=dict(source='generation_region.json',old_bounds_source='geometry.old_local_low/high, geometry.axes and geometry.origin',
            old_corners_world=old_world.tolist(),new_corners_world=new_world.tolist(),
            old_corners_pixels=old_xy.tolist(),new_corners_pixels=new_xy.tolist(),support_face_pixels=support.tolist()),
        masks=dict(edit='edit_mask.png',protection='protected_foreground_mask.png & extension_proposal.png',
            boundaries_unchanged=True,cyan_is_visualization_annotation=True),
        depth_contrast=contrast,statistics=previous_metrics,
        layout=dict(size_mm=[W,H],dpi=300,independent_panel_dpi=600,
            top_panel_size_mm=[panel_w,image_h],analysis_panel_size_mm=[panel_w,image_h],
            columns_x_mm=xs,foreground_inset_mm=[inset_x,inset_y,inset_size,inset_size],
            old_standalone_panels_removed=['Generated 2D Reference','Workspace Boundary','Foreground Preservation'],
            workspace_inset='Not repeated: all real 3D extents and support are combined in the scene panel'),
        qwen_calls=0,trellis_calls=0,new_m1_m4_runs=0,new_gaussian_renders=0))
    assert all(sha(PREVIOUS/name)==digest for name,digest in hashes.items())
    write(OUT/'validation.json',dict(previous_version_unchanged=True,source_files_byte_preserved=True,
        confirmed_f05_generation_hash_matches=True,same_camera_and_native_rgb_resolution=True,
        current_geometry_bbox_matches_saved_projection=True,old_extent_from_saved_region=True,
        masks_unchanged=True,protected_edit_overlap_pixels=0,
        all_edit_depth_pixels_black=True,contrast_display_only=True,
        inside_outside_metrics_identical_to_previous=True,condition_version_disclosure_present=True))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2000','-png',str(OUT/'figure_F11_revised.pdf'),str(OUT/'pdf_proof')],check=True)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>F11 revised · Case00</title>'
        '<style>body{max-width:1500px;margin:24px auto;font:16px/1.5 Arial}img{width:100%}</style>'
        '<h1>F11 · Case00 · Revised layout</h1><p>Current spatial conditions and the confirmed historical reference; no new experiment.</p>'
        '<p><a href="figure_F11_revised.png">300 DPI PNG</a> · <a href="figure_F11_revised.pdf">PDF</a> · '
        '<a href="panels/">Panels</a> · <a href="caption.txt">Caption</a> · <a href="sources.json">Sources</a></p>'
        '<img src="figure_F11_revised.png">',encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),inside_rgb_mae=inside,outside_rgb_mae=outside,
                          depth_contrast=contrast,old_files_preserved=True)),flush=True)


if __name__=='__main__':
    main()
