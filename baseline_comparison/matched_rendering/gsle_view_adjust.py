"""Camera-only GSLE revisions; preserve accepted GE, DGE and Ours images."""
import argparse
import copy
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import render as r
import localization_red as lr

BASE = r.BASE / 'target_centered_views'
REVISIONS = {'case07': 'case07_gsle_ours_view_v2',
             'case10': 'case10_gsle_wider_view_v2'}


@torch.no_grad()
def wider_camera(ref, ref_center, center, up, obj, scene, flags, original, loc_flags):
    """Keep the Ours camera orientation/intrinsics, transport to GSLE's target."""
    distance = float(np.linalg.norm(ref.center - ref_center))
    candidates = []
    for factor in (1.25, 1.5, 1.8, 2.1):
        cam = r.camera_at(ref, ref_center, center, distance * factor, up,
                          name='gsle_ours_direction_wider')
        stats = r.candidate_metrics(cam, obj, scene, flags, res=256)
        local = r.alpha_stats(r.c.raster(cam.resize(256, 256), original, loc_flags)[0].cpu().numpy())
        box = stats['bbox']
        margin = min(box[0], box[1], 256-box[2], 256-box[3]) if box else -256
        score = 50*max(0, .985-stats['capture']) + 8*max(0, .8-stats['visibility'])
        score += 1.5*abs(math.log(max(stats['size'], .001)/.48))
        score += 2*max(0, .7-stats['scene_coverage']) + 2*max(0, 1-local['pixels']/250)
        candidates.append(dict(camera=cam.record(), distance_factor=factor,
            distance=distance*factor, score=score, object=stats, localization=local,
            framed=stats['capture'] >= .985 and margin >= 8 and local['pixels'] >= 250))
    eligible = [x for x in candidates if x['framed']]
    assert eligible, 'No wider camera frames the actual GSLE target; inspect before publishing.'
    best = min(eligible, key=lambda x: x['score'])
    return r.c.Camera.from_record(best['camera']), dict(selected=best, candidates=candidates,
        policy='Exact Ours optical orientation, roll and intrinsics. Translate to the native GSLE target and increase distance. No Gaussian transform.')


def figures(out, case, report):
    caption = ('GSLE = Ours camera' if case == 'case07' else
               'GSLE: Ours direction, own target, wider view') + ' | GE / DGE / Ours unchanged'
    for stage in ('localization', 'final'):
        lr.board(out / f'comparison_{stage}.png', [[('Original', out/'ours_original.png', None)] +
            [(label, out/f'{method}_{stage}.png', None) for method, label in zip(lr.METHODS, lr.LABELS)]],
            case + ' | ' + stage, caption)
    for name, cropped in [('comparison_paired', False), ('comparison_crops', True)]:
        lr.board(out/(name+'.png'), [[(label+' / '+stage, out/f'{method}_{stage}.png',
            report['methods'][method]['shared_crop'] if cropped else None)
            for method, label in zip(lr.METHODS, lr.LABELS)] for stage in ('localization', 'final')],
            case + ' | Same-camera localization / final', caption)
    for n in (1, 2):
        lr.board(out/f'comparison_aux{n}.png', [[(label+' / '+stage,
            out/'views'/f'aux{n}'/f'{method}_{stage}.png', None)
            for method, label in zip(lr.METHODS, lr.LABELS)] for stage in ('localization', 'final')],
            case + f' | Auxiliary {n}', caption)
    explanation = ('GSLE 的主视角和两个辅助视角均完整复用 Ours 相机。' if case == 'case07' else
        'GSLE 与 Ours 的目标位置不同：保留 Ours 的观察方向、俯仰、内参，将相机移到 GSLE 自身目标附近并增加观察距离。两者并非同一世界坐标相机。')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{case} GSLE 视角调整</title>
    <style>body{{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}}img{{max-width:100%}}section{{margin:26px 0}}</style>
    <h1>{case}：GSLE 视角调整</h1><p>{explanation}</p><p>GaussianEditor、DGE、Ours 的图片和相机保持调整前的版本不变。每种方法的定位与最终结果共用相机；只重新渲染 GSLE，没有训练、修改分割或改动三维场景。GSLE 使用原始 Native DI-SDS 结果。</p>
    <p><a href="../index.html">全部案例</a> | <a href="camera_params.json">相机参数</a> | <a href="report.json">检查记录</a></p>'''
    for name, label in [('comparison_paired', '定位 / 最终结果'), ('comparison_final', '最终结果'),
        ('comparison_localization', '定位'), ('comparison_crops', '局部放大'),
        ('comparison_aux1', '辅助视角 1'), ('comparison_aux2', '辅助视角 2')]:
        page += f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
    (out/'index.html').write_text(page+'</html>', encoding='utf-8')


@torch.no_grad()
def run(case):
    previous = BASE/case
    out = BASE/REVISIONS[case]
    assert not (out/'report.json').exists(), 'Preserve completed display revision'
    out.mkdir(parents=True, exist_ok=True)
    spec = r.read(r.BASE/'prompts.json')
    row = next(x for x in spec['cases'] if x['case'] == case)
    report = r.read(previous/'report.json')
    cameras = r.read(previous/'camera_params.json')
    info = copy.deepcopy(r.read(r.BASE/'matched_views'/case/'camera_params.json')['methods']['gsle']['target'])
    paths = [Path(spec['source_scene']), Path(info['scene_path']), Path(info['object_path']),
             lr.index_path(row, 'gsle'), previous/'camera_params.json', previous/'report.json']
    hashes = {str(p): r.sha(p) for p in paths}
    copied = {}
    for p in previous.rglob('*.png'):
        if p.name.startswith(('gaussianeditor_', 'dge_', 'ours_')):
            relative = p.relative_to(previous)
            dest = out/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied[str(relative)] = r.sha(p)
    original = r.c.GaussianData.load(spec['source_scene'])
    scene = r.c.GaussianData.load(info['scene_path'])
    obj, vis_scene, flags = r.flag_data(info, scene, original)
    ids = np.load(lr.index_path(row, 'gsle'))
    idx = torch.as_tensor(ids, device='cuda')
    local_flags = torch.zeros_like(original.xyz)
    local_flags[idx] = 1
    ours_views = cameras['methods']['ours']['views']
    ours_center = np.array(cameras['methods']['ours']['target_center'])
    center = np.array(info['center'])
    up = r.c.normalized(np.array(r.read(row['target_record'])['up']))
    selections, selected_views = {}, {}
    for name, record in ours_views.items():
        ref = r.c.Camera.from_record(record['final_camera'])
        if case == 'case07':
            cam = ref
            selections[name] = dict(policy='Exact saved Ours camera', exact_ours_camera=True)
        else:
            cam, selections[name] = wider_camera(ref, ours_center, center, up, obj,
                                                vis_scene, flags, original, local_flags)
            assert np.array_equal(cam.R, ref.R)
        selected_views[name] = cam
    method = copy.deepcopy(report['methods']['gsle'])
    method.update(views={}, selection=selections, previous_selection=method['selection'])
    for name, cam in selected_views.items():
        folder = out/'views'/name
        folder.mkdir(parents=True, exist_ok=True)
        colors = original.colors(cam.center)
        base = r.c.raster(cam, original, colors)
        # Preserve this case's existing annotation style: red render color only.
        colors[idx] = colors.new_tensor([1., 0., 0.])
        red = r.c.raster(cam, original, colors)
        final = r.c.render(cam, scene, aux=False)['rgb']
        contribution = r.c.raster(cam, original, local_flags)[0]
        alpha = r.object_alpha(cam, obj)
        obj_stats = r.alpha_stats(alpha)
        loc_stats = r.alpha_stats(contribution.cpu().numpy())
        for suffix, image in [('original', base), ('localization', red), ('final', final),
            ('localization_mask', contribution[None]), ('object_alpha', torch.from_numpy(alpha)[None])]:
            r.c.save_image(folder/f'gsle_{suffix}.png', image)
        if name == 'main':
            for suffix in ('original', 'localization', 'final'):
                shutil.copy2(folder/f'gsle_{suffix}.png', out/f'gsle_{suffix}.png')
            boxes = [x for x in (obj_stats['bbox'], loc_stats['bbox']) if x]
            box = [min(x[0] for x in boxes), min(x[1] for x in boxes),
                   max(x[2] for x in boxes), max(x[3] for x in boxes)] if boxes else None
            method['shared_crop'] = r.square_crop(box, 1024)
        method['views'][name] = dict(camera_identical=True, exact_ours_camera=case == 'case07',
            object_projection=obj_stats, localization_projection=loc_stats)
        cameras['methods']['gsle']['views'][name] = dict(localization_camera=cam.record(),
            final_camera=cam.record(), original_camera=cam.record())
        print(case, name, 'rendered', flush=True)
    cameras['methods']['gsle']['selection'] = selections
    cameras['policy'] = ('GSLE exact Ours camera' if case == 'case07' else
        'GSLE uses Ours direction at its own target with increased distance') + '; GE / DGE / Ours unchanged; each localization/final pair shares a camera.'
    report['methods']['gsle'] = method
    report.update(revision=out.name, previous_revision=str(previous), input_sha256=hashes,
        preserved_image_sha256=copied, training_steps=0, ply_writes=0,
        ge_dge_ours_unchanged=True, source_files_unchanged=True)
    for p, h in hashes.items():
        assert r.sha(p) == h
    for p, h in copied.items():
        assert r.sha(out/p) == h
    r.save(out/'camera_params.json', cameras)
    figures(out, case, report)
    r.save(out/'report.json', report)
    with zipfile.ZipFile(out/(out.name+'.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.png', '.json', '.html'):
                if p.name == 'index.html':
                    archive.writestr('index.html', p.read_text(encoding='utf-8').replace('<a href="../index.html">全部案例</a> | ', ''))
                else:
                    archive.write(p, p.relative_to(out))
    print('COMPLETE', case, out, flush=True)


def publish():
    for case, revision in REVISIONS.items():
        assert (BASE/revision/'report.json').exists()
    # Preserve case04 (already exact GSLE/Ours) and the accepted case00/02 rows.
    page = (BASE/'index.html').read_text(encoding='utf-8')
    for case, revision in REVISIONS.items():
        page = page.replace(f'"{case}/', f'"{revision}/')
    page = page.replace('其余案例保持原样。', '')
    page = page.replace('每个方法朝向自己实际生成 / 定位的目标，不再优先贴近 Ours 方向。每个方法的红色定位与最终结果使用完全相同的相机。不同方法的观察方向和相机位置可以不同。仅在视角接近且能完整展示目标时复用 Ours 相机。',
        '按案例保存确认后的展示视角。每个方法的红色定位与最终结果使用完全相同的相机；跨方法是否共用相机以各案例说明为准。case00、02 全部共用 Ours 相机；case04、07 的 GSLE 共用 Ours 相机；case10 的 GSLE 对齐观察方向，但围绕自身目标拉远取景。')
    notice = '<p><strong>本次调整：case04、07 的 GSLE 共用 Ours 相机；case10 的 GSLE 保持 Ours 观察方向，朝向自身目标并拉远。三个案例的 GaussianEditor、DGE 保留原展示。</strong></p>'
    if notice not in page:
        page = page.replace('<h1>', notice+'<h1>', 1)
    for stage in ('localization', 'final'):
        old = f'overview_{stage}_case00_case02_case04_v4.png'
        new = f'overview_{stage}_gsle_04_07_10_v5.png'
        im = Image.open(BASE/old).convert('RGB')
        for case, y in [('case07', 1710), ('case10', 2250)]:
            part = Image.open(BASE/REVISIONS[case]/f'comparison_{stage}.png')
            # Only GSLE's column changes. All other methods remain pixel exact.
            im.paste(part.crop((1500, 80, 2000, 620)), (1500, y))
        im.save(BASE/new, dpi=(300, 300))
        page = page.replace(old, new)
    (BASE/'index.html').write_text(page, encoding='utf-8')
    print('PUBLISHED gallery', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=list(REVISIONS))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.case:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.45)
        run(args.case)
    if args.publish:
        publish()
