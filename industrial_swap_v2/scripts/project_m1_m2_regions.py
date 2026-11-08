"""Read-only Gaussian-mask / target-box / generation-box anchor diagnostic.

Run on the remote gsle environment. This does not execute any pipeline stage.
"""
import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import torch

from ..core.context import read_json, write_json, sha256


GREEN = (40, 245, 120)
ORANGE = (255, 163, 28)
MAGENTA = (248, 66, 220)


def tint(image, mask, color, amount):
    result = image.copy()
    result[mask] = np.round((1 - amount) * result[mask] + amount * np.asarray(color)).astype(np.uint8)
    return result


def boundary(image, mask, color, width=3):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, width, lineType=cv2.LINE_AA)


def wireframe(image, xy, color):
    # Both stored corner arrays enumerate the three binary corner coordinates.
    points = np.round(xy).astype(int)
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if i < j:
                cv2.line(image, tuple(points[i]), tuple(points[j]), color, 1, cv2.LINE_AA)


def font(size):
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size)


def figures(out, rgb, old, target, generation, target_xy, generation_xy, count, camera):
    overlay = tint(rgb, old, GREEN, .30)
    wireframe(overlay, generation_xy, MAGENTA)
    wireframe(overlay, target_xy, ORANGE)
    boundary(overlay, generation, MAGENTA, 4)
    boundary(overlay, target, ORANGE, 4)
    boundary(overlay, old, GREEN, 2)
    Image.fromarray(overlay).save(out / 'overlay_pixels.png')
    w, h = camera.width, camera.height
    main = Image.new('RGB', (w, h + 180), 'white')
    d = ImageDraw.Draw(main)
    d.text((20, 10), f'案例 01：三个区域投影到同一 Anchor View / {camera.name}', font=font(25), fill='#19212b')
    for y, color, label in (
        (54, GREEN, f'M1 实际旧设备 Mask：{count:,} 个选中高斯的可见贡献 > 0.02'),
        (90, ORANGE, 'M1 target_bbox：旧设备三维框投影'),
        (126, MAGENTA, 'M2 generation_bbox：新设备生成空间投影'),
    ):
        d.rectangle((20, y + 5, 44, y + 25), fill=color)
        d.text((56, y), label, font=font(22), fill='#19212b')
    main.paste(Image.fromarray(overlay), (0, 180))
    main.save(out / 'case01_anchor_overlay.png')

    panels = [(rgb, '原始 Anchor RGB', None)]
    for mask, color, title, xy in (
        (old, GREEN, 'M1 实际选中旧设备高斯 Mask', None),
        (target, ORANGE, 'M1 target_bbox 投影', target_xy),
        (generation, MAGENTA, 'M2 generation_bbox 投影', generation_xy),
    ):
        panel = tint(rgb, mask, color, .35)
        if xy is not None:
            wireframe(panel, xy, color)
        boundary(panel, mask, color)
        panels.append((panel, title, int(mask.sum())))
    size = 768
    sheet = Image.new('RGB', (2 * size, 2 * (size + 84) + 66), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((20, 12), '案例 01 | 同一相机、同一像素坐标；框内面积包含空白空间，不等于设备轮廓', font=font(27), fill='#19212b')
    for i, (panel, title, pixels) in enumerate(panels):
        x, y = (i % 2) * size, 66 + (i // 2) * (size + 84)
        d.text((x + 16, y + 7), title, font=font(27), fill='#19212b')
        subtitle = f'{camera.name} | {w} × {h}' if pixels is None else f'{pixels:,} px | 占全图 {100 * pixels / (w * h):.2f}%'
        d.text((x + 16, y + 46), subtitle, font=font(22), fill='#455363')
        sheet.paste(Image.fromarray(panel).resize((size, size), Image.Resampling.LANCZOS), (x, y + 84))
    sheet.save(out / 'case01_regions_separate.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli'))
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--m2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--old-alpha-threshold', type=float, default=.02)
    a = p.parse_args()
    if a.old_alpha_threshold != .02:
        raise ValueError('This diagnostic labels the current M4 old-mask threshold, 0.02')
    out = a.output.resolve()
    out.relative_to((a.root / 'industrial_swap_v2').resolve())
    out.mkdir(parents=True, exist_ok=False)
    os.environ['ROOT'] = str(a.root)
    os.environ['GSLE_OUTPUT'] = str(out / 'unused_adapter_output')
    os.environ['GSLE_RECORDS'] = str(out / 'unused_adapter_records')
    sys.path.insert(0, str(a.root / 'gsle'))
    import common as c

    m1 = a.case / 'outputs/m1'
    manifest = read_json(m1 / 'manifest.json')
    region = read_json(m1 / 'edit_region.json')
    camera_record = read_json(a.m2 / 'anchor_camera.json')
    camera = c.Camera.from_record(camera_record)
    inputs = [m1 / name for name in ('manifest.json', 'target.json', 'edit_region.json',
              'target_indices.npy', 'target_gaussians.ply', 'background_gaussians.ply')]
    inputs += [a.m2 / name for name in ('anchor_camera.json', 'anchor.png', 'edit_mask.png', 'conditions.json')]
    inputs.append(Path(manifest['source_scene']))
    hashes = {str(path): sha256(path) for path in inputs}
    assert hashes[manifest['source_scene']] == manifest['source_sha256']
    for name, expected in read_json(a.case / 'integrity.json').items():
        assert sha256(a.case / name) == expected, name
    target_corners = np.asarray(manifest['target_bbox'])
    generation_corners = np.asarray(manifest['generation_bbox'])
    np.testing.assert_array_equal(target_corners, read_json(m1 / 'target.json')['bbox_corners'])
    np.testing.assert_array_equal(generation_corners, region['bbox_corners'])
    target_xy, target_z = camera.project(target_corners)
    generation_xy, generation_z = camera.project(generation_corners)
    assert min(target_z.min(), generation_z.min()) > .01
    target_mask = c.bbox_mask(camera, target_corners).astype(bool)
    generation_mask = c.bbox_mask(camera, generation_corners).astype(bool)
    saved_mask = np.asarray(Image.open(a.m2 / 'edit_mask.png').convert('L')) > 127
    np.testing.assert_array_equal(generation_mask, saved_mask)

    ids = np.load(m1 / 'target_indices.npy')
    vertices = PlyData.read(manifest['source_scene'])['vertex'].data
    selected = PlyData.read(str(m1 / 'target_gaussians.ply'))['vertex'].data
    assert vertices.dtype == selected.dtype
    assert vertices[ids].tobytes() == selected.tobytes(), 'M1 target PLY differs from selected source vertices'
    del vertices, selected
    print(f'Rendering {len(ids)} actual selected Gaussians on {camera.name}', flush=True)
    scene = c.GaussianData.load(manifest['source_scene'])
    with torch.no_grad():
        flags = torch.zeros((len(scene.xyz), 3), device=scene.xyz.device)
        flags[torch.as_tensor(ids, device=scene.xyz.device)] = 1
        # Joint compositing includes all background occlusion, just as M4 M_old does.
        visible = c.raster(camera, scene, flags)[0].cpu().numpy()
        selected_scene = scene.subset(torch.as_tensor(ids, device=scene.xyz.device))
        independent = c.raster(camera, selected_scene, torch.ones_like(selected_scene.xyz))[0].cpu().numpy()
        rendered = c.render(camera, scene, aux=False)['rgb'].cpu().numpy().transpose(1, 2, 0)
        target_rgb = c.render(camera, selected_scene, aux=False)['rgb'].cpu().numpy().transpose(1, 2, 0)
    old_mask = visible > a.old_alpha_threshold
    isolated_mask = independent > a.old_alpha_threshold
    rgb = np.asarray(Image.open(a.m2 / 'anchor.png').convert('RGB'))
    rerender = np.round(np.clip(rendered, 0, 1) * 255).astype(np.uint8)
    rgb_delta = np.abs(rerender.astype(np.int16) - rgb.astype(np.int16))
    assert rgb.shape == (camera.height, camera.width, 3)
    assert rgb_delta.max() <= 1, 'Current renderer does not reproduce the stored Anchor'
    c.save_image(out / 'old_target_visible_alpha.png', visible)
    c.save_image(out / 'old_target_independent_alpha.png', independent)
    Image.fromarray(np.round(np.clip(target_rgb, 0, 1) * 255).astype(np.uint8)).save(out / 'old_target_independent_rgb.png')
    for name, mask in (('old_target_visible_mask', old_mask), ('old_target_independent_mask', isolated_mask),
                       ('target_bbox_mask', target_mask), ('generation_bbox_mask', generation_mask)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f'{name}.png')
    np.savez_compressed(out / 'gaussian_alpha.npz', visible=visible, independent=independent)
    write_json(out / 'anchor_camera.json', camera_record)
    figures(out, rgb, old_mask, target_mask, generation_mask, target_xy, generation_xy, len(ids), camera)
    after = {str(path): sha256(path) for path in inputs}
    assert hashes == after, 'An input changed during this diagnostic'
    report = {
        'case': str(a.case), 'm2': str(a.m2), 'output': str(out), 'command': sys.argv,
        'camera': camera_record, 'target_gaussians': len(ids), 'source_gaussians': len(scene.xyz),
        'old_mask_definition': 'Actual selected M1 Gaussians visible contribution in full original scene > 0.02; no dilation',
        'old_mask_threshold': a.old_alpha_threshold,
        'visible_target_pixels': int(old_mask.sum()), 'independent_target_pixels': int(isolated_mask.sum()),
        'target_bbox_pixels': int(target_mask.sum()), 'generation_bbox_pixels': int(generation_mask.sum()),
        'generation_to_target_box_area_ratio': float(generation_mask.sum() / target_mask.sum()),
        'generation_to_visible_target_area_ratio': float(generation_mask.sum() / old_mask.sum()),
        'generation_outside_target_box_pixels': int((generation_mask & ~target_mask).sum()),
        'visible_target_outside_target_box_pixels': int((old_mask & ~target_mask).sum()),
        'visible_target_outside_generation_box_pixels': int((old_mask & ~generation_mask).sum()),
        'target_bbox_corners': target_corners.tolist(), 'target_bbox_projected_uv': target_xy.tolist(),
        'generation_bbox_corners': generation_corners.tolist(), 'generation_bbox_projected_uv': generation_xy.tolist(),
        'original_oriented_side_lengths': region['original_oriented_side_lengths'],
        'generation_oriented_side_lengths': region['generation_oriented_side_lengths'],
        'generation_region_rule': region['generation_region_rule'],
        'source_selection_vertex_bytes_match': True, 'generation_projection_matches_existing_m2_mask': True,
        'anchor_rerender_max_uint8_difference': int(rgb_delta.max()),
        'anchor_rerender_mean_uint8_difference': float(rgb_delta.mean()),
        'input_files_unchanged': hashes == after, 'input_sha256': hashes,
        'note': 'Gaussian covariance support can extend beyond a bounding box derived from Gaussian centers.',
        'qwen_called': False, 'models_generated_or_trained': False,
    }
    write_json(out / 'projection_report.json', report)
    print({k: report[k] for k in ('visible_target_pixels', 'target_bbox_pixels', 'generation_bbox_pixels',
           'generation_to_target_box_area_ratio', 'anchor_rerender_max_uint8_difference', 'input_files_unchanged')}, flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    main()
