"""Display revision: translucent localization and a 20% wider case10 Ours rig.

Only annotation colors and camera positions change. No PLY is written.
"""
import argparse
import copy
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import render as r
import localization_red as lr
import publish_selected_comparison as publication

PREVIOUS = r.BASE/'comparison_updated_v6'
OUT = r.BASE/'comparison_updated_v7'
CASES = list(publication.SOURCES)
FACTOR = 1.20


def figures(case, report):
    out = OUT/case
    caption = 'Red annotation: 60% red + 40% original | Same camera within each column'
    if case == 'case10':
        caption += ' | Ours distance +20%'
    for stage in ('localization', 'final'):
        lr.board(out/f'comparison_{stage}.png', [[('Original', out/'ours_original.png', None)] +
            [(label, out/f'{m}_{stage}.png', None) for m, label in zip(lr.METHODS, lr.LABELS)]],
            case+' | '+stage, caption)
    for name, cropped in [('comparison_paired', False), ('comparison_crops', True)]:
        lr.board(out/(name+'.png'), [[(label+' / '+stage, out/f'{m}_{stage}.png',
            report['methods'][m]['shared_crop'] if cropped else None)
            for m, label in zip(lr.METHODS, lr.LABELS)] for stage in ('localization', 'final')],
            case+' | Same-camera localization / final', caption)
    for n in (1, 2):
        lr.board(out/f'comparison_aux{n}.png', [[(label+' / '+stage,
            out/'views'/f'aux{n}'/f'{m}_{stage}.png', None)
            for m, label in zip(lr.METHODS, lr.LABELS)] for stage in ('localization', 'final')],
            case+f' | Auxiliary {n}', caption)


@torch.no_grad()
def render_case(case, original, spec):
    out = OUT/case
    row = next(x for x in spec['cases'] if x['case'] == case)
    old_report = r.read(PREVIOUS/case/'report.json')
    cameras = r.read(PREVIOUS/case/'camera_params.json')
    report = dict(case=case, revision=OUT.name, previous_revision=str(PREVIOUS/case),
        annotation_red_opacity=.6, annotation_original_color_weight=.4,
        physical_gaussian_opacity_unchanged=True, training_steps=0, ply_writes=0,
        full_native_localization_indices=True, methods=copy.deepcopy(old_report['methods']))
    up = r.c.normalized(np.array(r.read(row['target_record'])['up']))
    paths = [lr.index_path(row, m) for m in lr.METHODS]
    scene = obj = None
    if case == 'case10':
        info = r.read(r.BASE/'matched_views'/case/'camera_params.json')['methods']['ours']['target']
        paths += [Path(info['scene_path']), Path(info['object_path'])]
        scene = r.c.GaussianData.load(info['scene_path'])
        obj, _, object_flags = r.flag_data(dict(info), scene, original)
    hashes = {str(p): r.sha(p) for p in paths}
    for method in lr.METHODS:
        idx = torch.as_tensor(np.load(lr.index_path(row, method)), device='cuda')
        loc_flags = torch.zeros_like(original.xyz)
        loc_flags[idx] = 1
        method_report = report['methods'][method]
        method_report['previous_view_metrics'] = method_report['views']
        method_report['views'] = {}
        method_report['annotation_red_opacity'] = .6
        changed_camera = case == 'case10' and method == 'ours'
        views = cameras['methods'][method]['views']
        for name, record in list(views.items()):
            before = r.c.Camera.from_record(record['final_camera'])
            cam = before
            if changed_camera:
                center = np.array(cameras['methods'][method]['target_center'])
                distance = float(np.linalg.norm(before.center-center))
                cam = r.camera_at(before, center, center, distance*FACTOR, up,
                                  name=before.name+'_distance120')
                assert np.array_equal(cam.R, before.R)
                record = dict(localization_camera=cam.record(), final_camera=cam.record(),
                              original_camera=cam.record())
                views[name] = record
            folder = out/'views'/name
            colors = original.colors(cam.center)
            source = r.c.raster(cam, original, colors)
            colors[idx] = .4*colors[idx]+.6*colors.new_tensor([1., 0., 0.])
            annotation = r.c.raster(cam, original, colors)
            contribution = r.c.raster(cam, original, loc_flags)[0]
            outside = contribution == 0
            error = float((annotation-source).abs()[:, outside].max()) if outside.any() else 0.
            assert error < 2e-6
            r.c.save_image(folder/f'{method}_localization.png', annotation)
            metrics = dict(localization_final_camera_identical=True,
                annotation_red_opacity=.6, nonselected_max_rgb_difference=error,
                camera_changed=changed_camera,
                localization_projection=r.alpha_stats(contribution.cpu().numpy()))
            if changed_camera:
                final = r.c.render(cam, scene, aux=False)['rgb']
                object_alpha = r.object_alpha(cam, obj)
                object_stats = r.alpha_stats(object_alpha)
                metrics.update(object_projection=object_stats, distance_before=distance,
                    distance_after=float(np.linalg.norm(cam.center-center)), distance_factor=FACTOR,
                    previous_camera=before.record(), orientation_unchanged=True)
                for suffix, image in [('original', source), ('final', final),
                    ('localization_mask', contribution[None]), ('object_alpha', torch.from_numpy(object_alpha)[None])]:
                    r.c.save_image(folder/f'{method}_{suffix}.png', image)
                if name == 'main':
                    for suffix in ('original', 'final'):
                        shutil.copy2(folder/f'{method}_{suffix}.png', out/f'{method}_{suffix}.png')
                    boxes = [b for b in [object_stats['bbox'], metrics['localization_projection']['bbox']] if b]
                    box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                           max(b[2] for b in boxes), max(b[3] for b in boxes)]
                    method_report['shared_crop'] = r.square_crop(box, cam.width)
                    metrics['visibility'] = r.candidate_metrics(cam, obj, scene, object_flags)
            if name == 'main':
                shutil.copy2(folder/f'{method}_localization.png', out/f'{method}_localization.png')
            method_report['views'][name] = metrics
        if changed_camera:
            cameras['methods'][method]['previous_selection'] = cameras['methods'][method].get('selection')
            cameras['methods'][method]['selection'] = dict(policy='User requested slightly wider Ours camera; distance +20%, same orientation and intrinsics.', distance_factor=FACTOR)
            method_report['previous_selection'] = method_report.get('selection')
            method_report['selection'] = cameras['methods'][method]['selection']
        print(case, method, 'complete', flush=True)
    cameras['policy'] = 'Preserve accepted cameras; case10 Ours main/aux distance +20%. Each localization/final pair shares a camera. Red annotation follows case00/02: 60% red + 40% original.'
    for p, digest in hashes.items():
        assert r.sha(p) == digest
    unchanged = {}
    for method in lr.METHODS:
        if case == 'case10' and method == 'ours':
            continue
        for p in (PREVIOUS/case).rglob(method+'_final.png'):
            rel = p.relative_to(PREVIOUS/case)
            assert r.sha(out/rel) == r.sha(p)
            unchanged[str(rel)] = r.sha(p)
    report.update(input_sha256=hashes, source_files_unchanged=True,
                  unchanged_final_sha256=unchanged)
    r.save(out/'prior_report.json', old_report)
    r.save(out/'report.json', report)
    r.save(out/'camera_params.json', cameras)
    figures(case, report)
    if scene is not None:
        del scene, obj
    torch.cuda.empty_cache()


def package(case04_source):
    notes = dict(publication.NOTES)
    notes['case04'] = 'GaussianEditor 采用指定 target_centered_views 页面当前的 case04 相机（与上一版相同）；分割与生成位置同时入镜。GSLE 与 Ours 共用相机。'
    notes['case07'] += ' 四种方法的定位均改为 60% 红色＋40% 原色，与 00、02 相同。'
    notes['case10'] += ' 四种方法的定位均改为相同半透明样式；Ours 观察距离增加 20%，方向和内参不变。'
    manifest = dict(revision=OUT.name, previous_revision=str(PREVIOUS), training_steps=0,
        ply_writes=0, annotation_red_opacity=.6, case10_ours_distance_factor=FACTOR,
        case04_ge_camera_source=str(case04_source), case04_ge_already_matched=True,
        cases={})
    for case in CASES:
        folder = OUT/case
        page = publication.header(case+' · 半透明标注更新版')
        page += f'<h1>{case} · 五案例更新版</h1><p class="note">{notes[case]}</p><p><a href="../index.html">五案例总览</a> | <a href="camera_params.json">相机记录</a> | <a href="report.json">检查记录</a></p><p>每列定位与最终结果共用相机。所有最终结果来自既有 Gaussian PLY；GSLE 为原始 Native DI-SDS。</p>'
        for name, label in publication.FIGURES:
            page += f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
        (folder/'index.html').write_text(page+'</html>', encoding='utf-8')
        manifest['cases'][case] = dict(explanation=notes[case], files_sha256={
            str(p.relative_to(folder)):r.sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix in ('.png', '.json')})
    for filename, suffix in [('overview_localization_final', 'paired'),
        ('overview_final', 'final'), ('overview_localization', 'localization')]:
        panels = [Image.open(OUT/case/f'comparison_{suffix}.png').convert('RGB') for case in CASES]
        canvas = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
        y = 0
        for panel in panels:
            canvas.paste(panel, (0, y));y += panel.height
        canvas.save(OUT/(filename+'.png'), dpi=(300, 300))
        ImageOps.contain(canvas, (1200, 4000)).save(OUT/(filename+'_web.jpg'), quality=94)
    page = publication.header('五案例更新版 · 半透明定位与取景微调')
    page += '<h1>五案例更新版：半透明定位与取景微调</h1><p class="note">00、02 原样保留。07、10 的全部红色定位标注与 00、02 统一；04 的 GaussianEditor 使用指定页面视角；10 的 Ours 观察距离增加 20%。只更新显示和相机，不修改三维模型。</p>'
    page += '<nav>'+' · '.join(f'<a href="#{case}">{case}</a>' for case in CASES)+'</nav>'
    page += '<p><a href="overview_localization_final.png">五案例高清总图</a> | <a href="overview_final.png">最终结果总图</a> | <a href="overview_localization.png">定位总图</a> | <a href="manifest.json">版本记录</a></p>'
    for case in CASES:
        page += f'<section id="{case}"><h2><a href="{case}/index.html">{case} · 完整结果</a></h2><p>{notes[case]}</p><a href="{case}/comparison_paired.png"><img src="{case}/comparison_paired.png" loading="lazy"></a></section>'
    page += '<p>每列上下为同相机；不同方法可使用不同相机位置，并非全部为同世界相机的像素级比较。</p></html>'
    (OUT/'index.html').write_text(page, encoding='utf-8')
    r.save(OUT/'manifest.json', manifest)
    with zipfile.ZipFile(OUT/(OUT.name+'.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png', '.jpg', '.json', '.html'):
                archive.write(p, p.relative_to(OUT))
    print('PACKAGED', OUT, flush=True)


@torch.no_grad()
def main():
    assert not OUT.exists(), 'Preserve existing display revisions'
    # The supplied gallery resolves case04 explicitly, avoiding stale assumptions.
    gallery = r.BASE/'target_centered_views'
    html = (gallery/'index.html').read_text(encoding='utf-8')
    match = re.search(r'href="(case04[^"/]*)/index.html"', html)
    assert match
    case04_source = gallery/match.group(1)
    source_params = r.read(case04_source/'camera_params.json')['methods']['gaussianeditor']
    previous_params = r.read(PREVIOUS/'case04/camera_params.json')['methods']['gaussianeditor']
    assert source_params == previous_params, 'Gallery changed; copy/re-render the requested GE camera before publishing.'
    for case in CASES:
        shutil.copytree(PREVIOUS/case, OUT/case)
    # Identical reference figures and final cameras for accepted case00/02/04.
    spec = r.read(r.BASE/'prompts.json')
    source_hash = r.sha(spec['source_scene'])
    original = r.c.GaussianData.load(spec['source_scene'])
    for case in ('case07', 'case10'):
        render_case(case, original, spec)
    assert r.sha(spec['source_scene']) == source_hash
    for case in ('case00', 'case02', 'case04'):
        for p in (PREVIOUS/case).rglob('*.png'):
            assert r.sha(p) == r.sha(OUT/case/p.relative_to(PREVIOUS/case))
    package(case04_source)


def publish_entry():
    assert (OUT/'manifest.json').exists() and (OUT/(OUT.name+'.zip')).exists()
    entry = PREVIOUS/'index.html'
    backup = PREVIOUS/'index_before_v7.html'
    if not backup.exists():
        shutil.copy2(entry, backup)
    page = (OUT/'index.html').read_text(encoding='utf-8')
    page = page.replace('<meta charset="utf-8">',
        '<meta charset="utf-8"><base href="../comparison_updated_v7/">', 1)
    entry.write_text(page, encoding='utf-8')
    print('UPDATED current entry', entry, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish_entry()
    else:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.4)
        main()
