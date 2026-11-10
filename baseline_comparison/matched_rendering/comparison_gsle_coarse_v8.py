"""Replace GSLE display renders with saved Coarse 6500 scenes, same cameras."""
import argparse
import copy
import shutil
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageOps

import render as r
import localization_red as lr
import publish_selected_comparison as publication

PREVIOUS = r.BASE/'comparison_updated_v7'
OUT = r.BASE/'comparison_updated_v8'
LABELS = ['GaussianEditor', 'DGE', 'GSLE Coarse', 'Ours']
NOTE = 'GSLE: saved Coarse 6500 (before DI-SDS) | All cameras and localization annotations unchanged'


def views(params, method):
    value = params['methods'][method]
    return value.get('views', value)


def figures(case, previous_report):
    out = OUT/case
    original = out/'original.png'
    if not original.exists():
        original = out/'ours_original.png'
    for stage in ('localization', 'final'):
        lr.board(out/f'comparison_{stage}.png', [[('Original', original, None)]+
            [(label, out/f'{m}_{stage}.png', None) for m, label in zip(lr.METHODS, LABELS)]],
            case+' | '+('Localization' if stage == 'localization' else 'Editing Results'), NOTE)
    def crop(method):
        return previous_report['methods'][method].get('shared_crop', previous_report.get('common_crop'))
    for filename, cropped in [('comparison_paired', False), ('comparison_crops', True)]:
        lr.board(out/(filename+'.png'), [[(
            label+' / '+('Localization' if stage == 'localization' else ('Render' if m == 'gsle' else 'Final')),
            out/f'{m}_{stage}.png', crop(m) if cropped else None)
            for m, label in zip(lr.METHODS, LABELS)] for stage in ('localization', 'final')],
            case+' | Same-camera localization / editing', NOTE)
    for n in (1, 2):
        lr.board(out/f'comparison_aux{n}.png', [[(
            label+' / '+('Localization' if stage == 'localization' else ('Render' if m == 'gsle' else 'Final')),
            out/'views'/f'aux{n}'/f'{m}_{stage}.png', None)
            for m, label in zip(lr.METHODS, LABELS)] for stage in ('localization', 'final')],
            case+f' | Auxiliary {n}', NOTE)


@torch.no_grad()
def render_case(case):
    previous = PREVIOUS/case
    out = OUT/case
    shutil.copytree(previous, out)
    params = r.read(previous/'camera_params.json')
    prior_report = r.read(previous/'report.json')
    native = r.BASE/'gsle_native'/case
    scene_path = native/'composition/scene.ply'
    object_path = native/'coarse/final_object_world.ply'
    progress_path = native/'records/coarse_progress.json'
    progress = r.read(progress_path)
    assert progress['stage'] == 'coarse' and progress['step'] == 6500
    input_hashes = {str(p):r.sha(p) for p in [scene_path, object_path, progress_path,
                                           previous/'camera_params.json']}
    scene = r.c.GaussianData.load(scene_path)
    # Verify that the scene really contains the saved Coarse object unchanged.
    info = dict(scene_path=str(scene_path), object_path=str(object_path))
    obj, _, _ = r.flag_data(info, scene, None)
    method_report = dict(stage='coarse', coarse_steps=6500, includes_di_sds=False,
        final_scene=str(scene_path), object_path=str(object_path),
        final_object_tail_verified=info['final_object_tail_verified'],
        object_gaussian_count=len(obj.xyz), views={})
    assert len(obj.xyz) == progress['points']
    for name, record in views(params, 'gsle').items():
        if not isinstance(record, dict) or 'final_camera' not in record:
            continue
        assert record['localization_camera'] == record['final_camera']
        cam = r.c.Camera.from_record(record['final_camera'])
        folder = out/'views'/name
        rgb = r.c.render(cam, scene, aux=False)['rgb']
        alpha = r.object_alpha(cam, obj)
        r.c.save_image(folder/'gsle_final.png', rgb)
        r.c.save_image(folder/'gsle_object_alpha.png', torch.from_numpy(alpha)[None])
        method_report['views'][name] = dict(camera_unchanged=True,
            localization_final_camera_identical=True, coarse_object_projection=r.alpha_stats(alpha))
        if name == 'main':
            shutil.copy2(folder/'gsle_final.png', out/'gsle_final.png')
        print(case, name, 'coarse rendered', flush=True)
    preserved = {}
    for p in previous.rglob('*.png'):
        rel = p.relative_to(previous)
        # Every other method image and every localization image is immutable.
        if p.name.startswith(('gaussianeditor_', 'dge_', 'ours_')) or p.name in (
                'original.png', 'gsle_localization.png', 'gsle_localization_mask.png', 'gsle_original.png'):
            digest = r.sha(p)
            assert r.sha(out/rel) == digest
            preserved[str(rel)] = digest
    assert r.sha(out/'camera_params.json') == input_hashes[str(previous/'camera_params.json')]
    for p, digest in input_hashes.items():
        assert r.sha(p) == digest
    report = dict(case=case, revision=OUT.name, previous_revision=str(previous),
        gsle_display_stage='coarse', gsle_display_steps=6500, training_steps=0, ply_writes=0,
        source_files_unchanged=True, all_cameras_unchanged=True,
        all_localization_images_unchanged=True, other_methods_unchanged=True,
        input_sha256=input_hashes, preserved_image_sha256=preserved,
        methods=copy.deepcopy(prior_report['methods']))
    report['methods']['gsle'] = method_report
    figures(case, prior_report)
    r.save(out/'report_before_coarse_display.json', prior_report)
    r.save(out/'report.json', report)
    del scene, obj
    torch.cuda.empty_cache()
    print('COMPLETE', case, 'Coarse 6500', method_report['object_gaussian_count'], flush=True)


def package():
    previous_manifest = r.read(PREVIOUS/'manifest.json')
    manifest = dict(revision=OUT.name, previous_revision=str(PREVIOUS),
        gsle_stage='coarse', gsle_steps=6500, gsle_includes_di_sds=False,
        training_steps=0, ply_writes=0, all_cameras_unchanged=True,
        all_localization_images_unchanged=True, other_methods_unchanged=True, cases={})
    for case in lr.CASES:
        folder = OUT/case
        note = previous_manifest['cases'][case]['explanation']
        page = publication.header(case+' · GSLE Coarse 对比')
        page += f'<h1>{case} · GSLE Coarse 对比</h1><p class="note">GSLE 结果替换为已保存的 Coarse 6500 步三维场景，尚未经过 DI-SDS 精细化。相机、定位图和其他三种方法保持上一版不变。</p><p>{note}</p><p><a href="../index.html">五案例总览</a> | <a href="camera_params.json">相机记录</a> | <a href="report.json">Coarse 来源与验证记录</a></p>'
        for name, label in publication.FIGURES:
            label = label.replace('最终结果', '编辑结果（GSLE 为 Coarse）')
            page += f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
        (folder/'index.html').write_text(page+'</html>', encoding='utf-8')
        manifest['cases'][case] = dict(explanation=note, report='report.json', files_sha256={
            str(p.relative_to(folder)):r.sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix in ('.png', '.json')})
    for filename, suffix in [('overview_localization_final', 'paired'),
        ('overview_final', 'final'), ('overview_localization', 'localization')]:
        panels = [Image.open(OUT/case/f'comparison_{suffix}.png').convert('RGB') for case in lr.CASES]
        canvas = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
        y = 0
        for panel in panels:
            canvas.paste(panel, (0, y));y += panel.height
        canvas.save(OUT/(filename+'.png'), dpi=(300, 300))
        ImageOps.contain(canvas, (1200, 4000)).save(OUT/(filename+'_web.jpg'), quality=94)
    page = publication.header('五案例更新版 · GSLE Coarse')
    page += '<h1>五案例对比：GSLE 改为 Coarse 6500 步</h1><p class="note">GSLE 列全部使用已保存的 Coarse 三维场景（DI-SDS 之前）。主视角、局部放大、两个辅助视角及总图均已更新。所有相机、半透明定位图、GaussianEditor、DGE 和 Ours 保持上一版不变。</p>'
    page += '<nav>'+' · '.join(f'<a href="#{case}">{case}</a>' for case in lr.CASES)+'</nav>'
    page += '<p><a href="overview_localization_final.png">五案例高清总图</a> | <a href="overview_final.png">编辑结果总图</a> | <a href="overview_localization.png">定位总图</a> | <a href="manifest.json">版本记录</a></p>'
    for case in lr.CASES:
        page += f'<section id="{case}"><h2><a href="{case}/index.html">{case} · 完整结果</a></h2><p>GSLE Coarse 6500 · 相机及定位图沿用上一版。</p><a href="{case}/comparison_paired.png"><img src="{case}/comparison_paired.png" loading="lazy"></a></section>'
    page += '<p>每列定位与结果共用相机；不同方法可使用不同相机位置。GSLE 列明确为粗生成阶段，不代表完整 DI-SDS 精细化结果。本次只读取既有 PLY 渲染，没有训练或修改三维模型。</p></html>'
    (OUT/'index.html').write_text(page, encoding='utf-8')
    r.save(OUT/'manifest.json', manifest)
    with zipfile.ZipFile(OUT/(OUT.name+'.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png', '.jpg', '.json', '.html'):
                archive.write(p, p.relative_to(OUT))
    print('PACKAGED', OUT, flush=True)


def publish():
    assert (OUT/'manifest.json').exists() and (OUT/(OUT.name+'.zip')).exists()
    active = r.BASE/'comparison_updated_v6/index.html'
    backup = active.with_name('index_before_v8.html')
    if not backup.exists():
        shutil.copy2(active, backup)
    page = (OUT/'index.html').read_text(encoding='utf-8').replace('<meta charset="utf-8">',
        '<meta charset="utf-8"><base href="../comparison_updated_v8/">', 1)
    active.write_text(page, encoding='utf-8')
    print('UPDATED current five-case entry', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
    else:
        assert not OUT.exists(), 'Preserve completed display revisions'
        OUT.mkdir(parents=True)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.4)
        for case in lr.CASES:
            render_case(case)
        package()
