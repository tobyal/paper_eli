"""Publish F09 using actual 2,000-step dual-mask scenes, never a silent Step-0 fallback."""
import argparse
import gc
import html
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from figure_common import ROOT, BASE, OUT, read, save, sha, source, export, plt
sys.path.insert(0, str(ROOT))
from industrial_swap_v2.core.context import Context, gsle

F09 = OUT / 'F09_text_control'
DEST = F09 / 'm4_2000_v2'
LABELS = ['Six-Axis Robot', 'Requested SCARA Robot', 'CNC Machining Center', 'Roller Conveyor']


def link(path, target):
    if path.exists():
        assert path.resolve() == target.resolve(), str(path)
    else:
        path.symlink_to(target.resolve())


def render_and_plot():
    previous = read(F09 / 'placement_results.json')
    old = Path(read(ROOT / 'baseline_comparison/prompts.json')['cases'][0]['ours_source'])
    original_f05 = read(OUT.parent / 'paper_figures_v3/F05_case00/sources.json')
    c = gsle(Context(old / 'config.yaml'))
    main_camera = c.Camera.from_record(previous['main_camera'])
    crop = previous['common_crop']
    rows, records = [], []
    for i, placed in enumerate(previous['variants']):
        key = placed['variant']
        dest = DEST / 'variants' / key
        dest.mkdir(parents=True, exist_ok=True)
        run = old if key == 'A_robot' else dest
        if key != 'A_robot':
            assert read(run / 'completed.json')['steps'] == 2000
        stage = run / 'm4'
        report = read(stage / 'summary.json')
        training = read(stage / 'background_optimization/summary.json')
        assert training['iterations'] == 2000
        assert training['source_hashes']['background'] == placed['background_sha256']
        assert report['candidate_sha256']['object'] == placed['object_sha256']
        assert all(report['checks'][k] for k in ('object_file_sha256_unchanged',
            'object_scene_rows_bitwise_identical', 'unselected_background_bitwise_identical',
            'background_geometry_bitwise_identical'))
        scene_path = Path(report['candidate_scene'])
        assert sha(scene_path) == report['candidate_sha256']['scene']
        if key == 'A_robot':
            assert sha(scene_path) == original_f05['final_scene']['sha256']
        paths = {'scene_step2000.ply': scene_path,
                 'background_step2000.ply': Path(report['candidate_background']),
                 'scene_guarded.ply': Path(report['guarded_scene']),
                 'object_world.ply': stage / 'object_world.ply'}
        for name, target in paths.items():
            link(dest / name, target)
        cameras = read(F09 / 'variants' / key / 'camera_params.json')
        save(dest / 'camera_params.json', cameras)
        shutil.copy2(stage / 'summary.json', dest / 'optimization_report.json')
        shutil.copy2(stage / 'transform.json', dest / 'transform.json')
        # Actual full-scene Gaussian rendering; no 2D repair or re-registration.
        scene = c.GaussianData.load(scene_path)
        with torch.no_grad():
            c.save_image(dest / 'main_step2000.png', c.render(main_camera, scene, aux=False)['rgb'])
            anchor = c.Camera.from_record(cameras['anchor'])
            c.save_image(dest / 'anchor_step2000.png', c.render(anchor, scene, aux=False)['rgb'])
        del scene
        gc.collect()
        torch.cuda.empty_cache()
        shutil.copy2(F09 / 'variants' / key / 'main_scene.png', dest / 'main_step0.png')
        for step in (0, 2000):
            Image.open(dest / f'main_step{step}.png').crop(crop).save(dest / f'detail_step{step}.png')
        if report['selected_step'] == 0:
            shutil.copy2(dest / 'main_step0.png', dest / 'main_guarded.png')
        else:
            guarded = c.GaussianData.load(report['guarded_scene'])
            with torch.no_grad():
                c.save_image(dest / 'main_guarded.png', c.render(main_camera, guarded, aux=False)['rgb'])
            del guarded
            torch.cuda.empty_cache()
        # Readable per-variant same-camera before/after proof.
        images = [Image.open(dest / f'detail_step{s}.png').convert('RGB').resize((700, 700)) for s in (0, 2000)]
        panel = Image.new('RGB', (1400, 745), 'white')
        draw = ImageDraw.Draw(panel)
        for j, im in enumerate(images):
            panel.paste(im, (700*j, 45))
        draw.text((12, 14), 'Before: placement only', fill='black')
        draw.text((712, 14), 'After: actual dual-mask M4 iteration 2000', fill='black')
        panel.save(dest / 'before_after.png')
        before = report['baseline']; after = report['candidate']
        row = {'variant': key, 'M4_background_steps': 2000, 'reused_existing_optimized_run': key == 'A_robot',
               'candidate_guard_passed': report['candidate_guard_passed'], 'guarded_selected_step': report['selected_step'],
               'anchor_visibility_before': before['anchor']['visibility'],
               'anchor_visibility_after': after['anchor']['visibility'],
               'validation_trusted_visibility_before': before['validation']['trusted_visibility'],
               'validation_trusted_visibility_after': after['validation']['trusted_visibility'],
               'anchor_outside_union_mae': after['anchor']['outside_union_mae'],
               'validation_environment_l1_before': before['validation']['environment_l1'],
               'validation_environment_l1_after': after['validation']['environment_l1'],
               'object_sha256': placed['object_sha256'], 'scene_sha256': report['candidate_sha256']['scene'],
               'original_background_sha256': placed['background_sha256'],
               'rgb_points': report['rgb_points'], 'occlusion_points': report['occlusion_points'],
               'alignment_gate_passed': placed['transform'].get('quality_gate_passed', True)}
        rows.append(row)
        records.append({'variant': key, 'training_run': str(run), 'rendered_scene': source(scene_path),
                        'object': source(stage / 'object_world.ply'), 'report': source(stage / 'summary.json'),
                        'metrics': row, 'checks': report['checks'], 'transform': placed['transform']})
        assert sha(F09 / 'variants' / key / 'scene_final.ply') == placed['scene_sha256']
        print('Rendered actual M4', key, flush=True)

    (DEST / 'panels').mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(10, 6.5))
    fig.subplots_adjust(left=.025, right=.98, bottom=.15, top=.87, wspace=.035, hspace=.12)
    fig.suptitle('Text-Controllable Industrial Equipment Replacement', y=.97, fontsize=14, fontweight='bold')
    for i, r in enumerate(records):
        for j, name in enumerate(('main_step2000.png', 'detail_step2000.png')):
            ax[j, i].imshow(Image.open(DEST / 'variants' / r['variant'] / name))
            ax[j, i].axis('off')
            if j == 0:
                ax[j, i].set_title(LABELS[i], fontsize=9)
    fig.text(.025, .06,
             'All columns: actual dual-mask background optimization at iteration 2,000; fixed devices and poses.\n'
             'Identical camera and crop. Candidates and guard-selected exports are recorded separately.', fontsize=8)
    failed = [r['variant'] for r in rows if not r['candidate_guard_passed']]
    caption = (
        'Figure 9. Text-controlled equipment replacement at one Case00 workstation after dual-mask local background optimization. '
        'All four columns render the actual 2000-step scenes in the identical camera; the bottom row uses the same pixel crop. '
        'The yellow robot reuses exactly the optimized scene used in F05; SCARA-request, CNC and roller-conveyor variants each run '
        '2000 new background-only steps from the same original M1 retained background, using their own per-view old/new masks. '
        'Environment RGB supervision excludes the union of old and new silhouettes; visibility supervision only updates eligible '
        'background opacity, and RGB updates only eligible background color/opacity. All device Gaussian parameters and their '
        'previous placements remain unchanged. No Qwen/TRELLIS generation, registration or floor completion was repeated. '
        'The SCARA request has an articulated-arm appearance; CNC foreground extraction and previous registration limitations persist. '
        'The displayed states are actual iteration-2000 candidates, not silently substituted guard-selected exports. '
        f'Candidates failing the original preservation guard: {", ".join(failed) if failed else "none"}. '
        'Guarded scenes and exact metrics are delivered separately; no guard thresholds were relaxed.')
    manifest = {'variants': records, 'main_camera': previous['main_camera'], 'common_crop': crop,
                'shared_initial_background': previous['shared_background'], 'backgrounds_individually_optimized': True,
                'display_policy': 'actual_iteration_2000_not_guard_fallback',
                'robot_matches_F05_scene_sha256': original_f05['final_scene']['sha256'],
                'qwen_calls': 0, 'trellis_calls': 0, 'object_and_pose_unchanged': True,
                'original_placement_outputs_preserved': True}
    save(DEST / 'optimization_results.json', manifest)
    export(fig, DEST, 'F09', caption, manifest, rows, ax.ravel())
    # Additional visual comparison only, not another paper figure.
    board = Image.new('RGB', (1400, 745*4), 'white')
    for i, r in enumerate(records):
        board.paste(Image.open(DEST / 'variants' / r['variant'] / 'before_after.png'), (0, 745*i))
    board.save(DEST / 'before_after_all.png')
    lines = ['F09 correction: actual dual-mask M4 at 2000 steps', '',
             'A_robot reuses exactly the existing F05 optimized scene; B/C/D run new 2000-step optimization.',
             'All objects, poses, camera and crop are fixed. The common initial M1 background is optimized separately per object.',
             'No Qwen, TRELLIS, registration or floor completion was rerun.', '',
             'Displayed scene_step2000.ply is the actual optimized candidate; scene_guarded.ply is the unchanged guard policy export.', '']
    for r in rows:
        lines.append(f"{r['variant']}: anchor visibility {r['anchor_visibility_before']:.4f} -> {r['anchor_visibility_after']:.4f}; "
                     f"validation trusted visibility {r['validation_trusted_visibility_before']:.4f} -> {r['validation_trusted_visibility_after']:.4f}; "
                     f"guard={r['candidate_guard_passed']}; guarded step={r['guarded_selected_step']}.")
    (DEST / 'report.txt').write_text('\n'.join(lines) + '\n')


def publish():
    # Preserve the placement-only publication files, then make the main review page show the correction.
    archive = F09 / 'placement_only_publication'
    names = ['figure_F09.png', 'figure_F09.pdf', 'preview_F09.jpg', 'caption.txt', 'sources.json', 'data.csv']
    if not archive.exists():
        archive.mkdir()
        for name in names:
            shutil.copy2(F09 / name, archive / name)
        shutil.copytree(F09 / 'panels', archive / 'panels')
    for name in names:
        shutil.copy2(DEST / name, F09 / name)
    shutil.copytree(DEST / 'panels', F09 / 'panels', dirs_exist_ok=True)
    root_index = OUT / 'index.html'
    page = root_index.read_text()
    page = page.replace('<a href="F09_generated_assets.zip">F09 三维结果压缩包</a>',
        '<a href="F09_generated_assets.zip">F09 原始生成资产</a> · '
        '<a href="F09_m4_2000_scenes.zip">F09 双 Mask 优化场景</a>')
    start = page.index('<section id="F09">')
    end = page.index('</section>', start) + len('</section>')
    section = ('<section id="F09"><h2>F09 · 双 Mask 优化 2000 步</h2>'
               '<p>机械臂使用与 F05 一致的已优化场景；其余三种设备完成各自的 2000 步背景优化。'
               '设备模型及位姿不变。这里展示实际第 2000 步候选，检查状态另行记录。</p>'
               '<a href="F09_text_control/figure_F09.png"><img src="F09_text_control/preview_F09.jpg?v=m4_2000" loading="lazy"></a>'
               '<p><a href="F09_text_control/figure_F09.pdf">PDF</a> · '
               '<a href="F09_text_control/figure_F09.png">300 DPI PNG</a> · '
               '<a href="F09_text_control/m4_2000_v2/index.html">优化前后及运行记录</a> · '
               '<a href="F09_text_control/data.csv">Data CSV</a> · '
               '<a href="F09_text_control/caption.txt">Caption</a> · '
               '<a href="F09_text_control/sources.json">Sources</a></p><p>' + html.escape((DEST/'caption.txt').read_text()) + '</p></section>')
    root_index.write_text(page[:start] + section + page[end:])
    cards = []
    for r in read(DEST / 'optimization_results.json')['variants']:
        key = r['variant']; m = r['metrics']
        cards.append(f'<section><h2>{html.escape(key)}</h2><p>2000 steps; guard={m["candidate_guard_passed"]}; '
                     f'guarded step={m["guarded_selected_step"]}; Anchor visibility '
                     f'{m["anchor_visibility_before"]:.3f} → {m["anchor_visibility_after"]:.3f}</p>'
                     f'<img src="variants/{key}/before_after.png"><p>'
                     f'<a href="variants/{key}/scene_step2000.ply">2000步完整场景 PLY</a> · '
                     f'<a href="variants/{key}/object_world.ply">固定设备 PLY</a> · '
                     f'<a href="variants/{key}/optimization_report.json">完整检查记录</a></p></section>')
    (DEST / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>F09 dual-mask M4</title>'
        '<style>body{max-width:1400px;margin:24px auto;font-family:Arial;background:#f4f6f8;padding:24px}section{background:white;padding:20px;margin:24px 0}img{width:100%}a{color:#195e9e}</style>'
        '<h1>F09 · 双 Mask 局部背景优化，2000 步</h1><p>左：仅放置；右：实际优化 2000 步。机械臂复用 F05 场景。</p>'
        '<p><a href="../../index.html#F09">返回论文图表</a> · '
        '<a href="../../F09_m4_2000_update.zip">更新图表包</a> · '
        '<a href="../../F09_m4_2000_scenes.zip">优化后三维场景包</a> · <a href="figure_F09.pdf">PDF</a></p>'
        '<img src="preview_F09.jpg">' + ''.join(cards))
    # Prepend a correction to the earlier report instead of erasing original execution history.
    summary = OUT / 'experiment_summary.md'
    original = summary.read_text()
    marker = '## F09 correction: dual-mask M4, 2026-11-11'
    if marker not in original:
        summary.write_text(marker + '\n\nThe previous placement-only F09 has been superseded by '
            '`F09_text_control/m4_2000_v2`. A uses the same optimized scene as F05; B/C/D each run 2000 new dual-mask background steps. '
            'All device parameters and poses are unchanged. The paper image renders actual iteration-2000 candidates; guard-selected '
            'scenes and failed checks remain separately recorded. The previous report below describes the original delivery.\n\n' + original)
    validation = read(OUT / 'delivery_validation.json')
    for r in validation['figures']:
        if r['figure'] == 'F09':
            r['png'] = source(F09 / 'figure_F09.png'); r['pdf'] = source(F09 / 'figure_F09.pdf')
            r['correction'] = 'dual-mask M4 2000, actual candidates'
    save(OUT / 'delivery_validation.json', validation)
    # Keep a previous publication ZIP and update only F09 plus the manifest/index/summary and new scripts.
    publication = OUT / 'paper_figures_v4_publication.zip'
    previous_zip = OUT / 'paper_figures_v4_publication_before_f09_m4.zip'
    if not previous_zip.exists():
        shutil.copy2(publication, previous_zip)
    changed = [*[(F09 / n) for n in names], *list((F09/'panels').glob('*.png')),
               OUT/'index.html', OUT/'experiment_summary.md', OUT/'delivery_validation.json']
    for p in DEST.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.png','.jpg','.pdf','.json','.csv','.txt','.html','.yaml'):
            continue
        rel = p.relative_to(DEST)
        if not any(x in rel.parts for x in ('m1','m2','m3','m4')):
            changed.append(p)
    for script in ('refine_f09_m4.py', 'finish_f09_m4.py', 'figure_f09.py'):
        target = OUT / 'reproduction_scripts' / script
        shutil.copy2(BASE / 'scripts/paper_figures_v4' / script, target)
        changed.append(target)
    changed_names = {str(p.relative_to(OUT)) for p in changed}
    temporary = publication.with_suffix('.tmp.zip')
    with zipfile.ZipFile(previous_zip) as oldzip, zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as newzip:
        for entry in oldzip.infolist():
            if entry.filename not in changed_names:
                newzip.writestr(entry, oldzip.read(entry.filename))
        for p in changed:
            newzip.write(p, p.relative_to(OUT))
    temporary.replace(publication)
    # Review material is small; huge checkpoints and per-iteration snapshots remain on the server.
    with zipfile.ZipFile(OUT/'F09_m4_2000_update.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in DEST.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.png','.jpg','.pdf','.json','.csv','.txt','.html','.yaml'):
                rel = p.relative_to(DEST)
                if 'm4' in rel.parts or any(x in rel.parts for x in ('m1','m2','m3')):
                    continue
                z.write(p, rel)
        for s in ('refine_f09_m4.py', 'finish_f09_m4.py', 'figure_f09.py'):
            z.write(BASE/'scripts/paper_figures_v4'/s, 'scripts/'+s)
    with zipfile.ZipFile(OUT/'F09_m4_2000_scenes.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for r in read(DEST / 'optimization_results.json')['variants']:
            p = DEST/'variants'/r['variant']
            for name in ('scene_step2000.ply','object_world.ply','camera_params.json','transform.json','optimization_report.json'):
                z.write(p/name, (p/name).relative_to(DEST))
        z.write(DEST/'optimization_results.json','optimization_results.json')
        z.write(DEST/'report.txt','report.txt')
    print('F09 DUAL-MASK PUBLICATION COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    if not args.publish_only:
        render_and_plot()
    publish()
