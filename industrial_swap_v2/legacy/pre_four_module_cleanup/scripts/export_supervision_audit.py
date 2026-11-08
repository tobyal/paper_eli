"""Read-only audit of actual cached supervision. No rendering/training/API calls.

PNG panels visualize existing tensors; masked RGB uses black only to show pixels
whose loss weight is zero. It is not a newly generated inpainting target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def png(path, values):
    values = np.asarray(values)
    Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype('uint8')).save(path)


def panel(path, images, labels, subtitles, width, title, footer):
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    large = ImageFont.truetype(font_path, 28)
    height = round(images[0].height * width / images[0].width)
    gap, top, header = 16, 70, 70
    board = Image.new('RGB', (width * 2 + gap * 3, top + (height + header + gap) * 2 + 70), '#eef1f4')
    draw = ImageDraw.Draw(board)
    draw.text((gap, 15), title, font=large, fill='#152333')
    for i, (image, label, subtitle) in enumerate(zip(images, labels, subtitles)):
        x = gap + (i % 2) * (width + gap)
        y = top + (i // 2) * (height + header + gap)
        draw.rectangle((x, y, x + width, y + header + height), fill='white')
        draw.text((x + 10, y + 6), label, font=font, fill='#152333')
        draw.text((x + 10, y + 38), subtitle, font=small, fill='#455468')
        board.paste(image.convert('RGB').resize((width, height), Image.Resampling.NEAREST if image.mode == 'L' else Image.Resampling.LANCZOS), (x, y + header))
    draw.text((gap, board.height - 58), footer, font=small, fill='#243347')
    board.save(path)


def export(run, out):
    out.mkdir(parents=True, exist_ok=False)
    folder = run / 'm3/multiview_visibility'
    manifest = read(folder / 'manifest.json')
    effective = read(run / 'm4/effective_config.json')
    candidates = read(folder / 'occlusion_candidates.json')
    source_hashes = {}
    for name in ('object_mask.png', 'object_white.png', 'object_rgba.png', 'anchor_camera.json'):
        source = run / 'm2' / name
        source_hashes[str(source.resolve())] = digest(source)
    assert digest(run / 'm2/object_mask.png') == manifest['source_hashes']['mask']
    assert digest(run / 'm2/anchor_camera.json') == manifest['source_hashes']['camera']

    # Same image conversion and interpolation as modules/m4_refinement.py.
    device = out / 'device_reference'
    device.mkdir()
    resolution = effective['config']['m4']['resolution']
    tensors = {}
    for name, mode, interpolation, filename in (
        ('rgb', 'RGB', 'bilinear', 'object_white.png'),
        ('mask', 'L', 'nearest', 'object_mask.png'),
    ):
        a = np.asarray(Image.open(run / 'm2' / filename).convert(mode)).copy() / 255
        if a.ndim == 2:
            a = a[:, :, None]
        x = torch.tensor(a.transpose(2, 0, 1), dtype=torch.float32)[None]
        kwargs = {'align_corners': False} if interpolation == 'bilinear' else {}
        x = F.interpolate(x, (resolution, resolution), mode=interpolation, **kwargs)[0].numpy()
        np.save(device / f'gt_{name}_chw.npy', x)
        a = x.transpose(1, 2, 0)
        png(device / f'gt_{name}_{resolution}.png', a[:, :, 0] if a.shape[-1] == 1 else a)
        tensors[name] = x
    shutil.copy2(run / 'm2/anchor_camera.json', device / 'camera.json')

    views = []
    for record in manifest['views']:
        cache_path = folder / 'cache' / (record['name'] + '.npz')
        with np.load(cache_path) as cache:
            target = out / 'views' / record['name']
            target.mkdir(parents=True)
            rgb = cache['original_rgb']
            mask = cache['background_train_mask']
            unknown = cache['unknown_region']
            Image.fromarray(rgb).save(target / 'real_photo_target.png')
            for key in ('background_train_mask', 'visibility_weight', 'object_alpha', 'unknown_region', 'legal_foreground'):
                png(target / (key + '.png'), cache[key])
            np.savez_compressed(target / 'loss_tensors.npz', original_rgb=rgb,
                background_train_mask=mask, visibility_weight=cache['visibility_weight'], object_alpha=cache['object_alpha'])
            Image.fromarray(rgb * mask[:, :, None]).save(target / 'supervised_rgb_display_only.png')
            overlay = rgb.astype(float)
            overlay[mask] = overlay[mask] * .55 + np.array([0, 200, 100]) * .45
            overlay[unknown] = overlay[unknown] * .5 + np.array([240, 60, 60]) * .5
            Image.fromarray(np.round(overlay).astype('uint8')).save(target / 'mask_overlay_display_only.png')
            entry = dict(record, cache_sha256=digest(cache_path),
                visibility_pixels_inside_unknown=int((cache['visibility_weight'] & unknown).sum()))
            views.append(entry)
            (target / 'camera_and_usage.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2))

    anchor = manifest['views'][0]
    anchor_dir = out / 'views' / anchor['name']
    panel(out / 'device_supervision.png', [
        Image.open(device / f'gt_rgb_{resolution}.png'),
        Image.open(device / f'gt_mask_{resolution}.png'),
        Image.open(anchor_dir / 'object_alpha.png'),
        Image.open(anchor_dir / 'visibility_weight.png')],
        ['B：设备 RGB 监督图', 'B：设备轮廓监督 Mask', 'A：期望设备可见贡献', 'A：遮挡损失实际生效区域'],
        ['M2 白底设备图，实际训练尺寸 512×512', 'M2 / SAM Mask；仅比较独立设备 Alpha',
         'M3 设备独立渲染 Alpha，训练尺寸 640×640', '白色参与可见性损失；黑色权重为 0'],
        512, '设备监督：同一 Anchor 相机，实际输入张量',
        'A = 背景优化；B = 设备外观优化。图像仅为导出展示，未重新生成设备。')

    R = np.asarray(anchor['camera']['R']); center = -R.T @ np.asarray(anchor['camera']['t'])
    scale = manifest['region_scale']
    def distance(v):
        rv = np.asarray(v['camera']['R']); cv = -rv.T @ np.asarray(v['camera']['t'])
        return np.linalg.norm(center - cv) / scale + np.linalg.norm(R - rv)
    ordered = sorted([v for v in views if v['split'] == 'train'], key=distance)
    selected = [ordered[0], ordered[len(ordered) // 2]]
    for i, record in enumerate(selected):
        d = out / 'views' / record['name']
        panel(out / ('environment_supervision.png' if i == 0 else 'environment_supervision_other_view.png'),
            [Image.open(d / name) for name in ('real_photo_target.png', 'background_train_mask.png',
                'supervised_rgb_display_only.png', 'mask_overlay_display_only.png')],
            ['A：真实视频帧监督图', 'A：环境 RGB 损失 Mask', '实际参与环境 RGB 损失的像素', '监督范围叠加说明'],
            [record['camera']['name'] + ' · 训练相机', '白色参与；黑色排除',
             '黑色只是零权重显示，不是补全目标颜色', '绿色：参与监督；红色：旧设备 / 未知排除区'],
            640, '环境监督：' + record['name'] + '（四图为同一相机）',
            '目标来自原视频，仍含旧设备；旧设备 / 未知区被排除，没有完整地面修复目标图。')

    diffusion = out / 'saved_diffusion_masks'
    shutil.copytree(run / 'm4/diffusion_masks', diffusion)
    # These are historical masks, not invented RGB targets or replayed teacher outputs.
    report = {'run': str(run), 'device_reference': str(device), 'source_hashes': source_hashes,
        'device_target_resolution': resolution, 'tensor_reconstruction': 'Same conversion/interpolation as M4, replayed on CPU; PNG is 8-bit display, NPY keeps float32.',
        'anchor': views[0], 'environment_panels': [v['name'] for v in selected],
        'train_views': sum(v['split'] == 'train' for v in views),
        'validation_views': sum(v['split'] == 'validation' for v in views),
        'candidate_count': candidates['candidate_count'],
        'optimized_candidates': candidates['supported_candidate_count'],
        'frozen_candidates': candidates['candidate_count'] - candidates['supported_candidate_count'],
        'optimized_background_count': candidates['local_supported_count'],
        'frozen_local_background_count': candidates['local_unsupported_frozen_count'],
        'stage_b_frozen': effective['frozen_parameters'], 'stage_b_trainable': effective['optimized_parameters'],
        'stage_b_reference_loss': effective['reference_loss'],
        'diffusion': {'fixed_rgb_target_exists': False,
            'actual_condition': 'Joint RGB / depth / dilated per-view object Alpha / text; frozen diffusion predicts a latent gradient.',
            'saved_artifacts': 'Only first-iteration four masks were saved, not per-step RGB/depth/noise/teacher prediction.',
            'historical_denoised_target_not_reconstructed': True},
        'notes': ['Anchor has no real-photo RGB supervision; its cached RGB is an unused all-zero placeholder.',
            'Train and validation folders are both exported; only train contributes to RGB optimization.',
            'Masked images and overlays are explanatory visualizations of cached weights, not new supervision.',
            'No original experiment, PLY, model parameters, or masks were modified.'],
        'background_support_diagnostic': read(run / 'validation/background_support_report.json')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('environment_panels','train_views','validation_views','candidate_count','optimized_candidates','frozen_candidates')}, indent=2))
    print('Exported:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())
