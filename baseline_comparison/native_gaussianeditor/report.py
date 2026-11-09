"""Export an offline, stage-by-stage native GE inspection page and summary."""
import html
import argparse
import json
import time
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'
OUT = BASE / 'gaussianeditor_native'
STAGES = [
    ('原始场景', ['original_view.png']),
    ('GE 原生定位', ['localization_mask_overlay.png', 'localization_render_selected.png',
                    'localization_scene_with_box.png', 'localization_views.png']),
    ('原生 Delete', ['delete_before.png', 'delete_target_mask.png', 'delete_after.png', 'delete_comparison.png']),
    ('GE 自身定位构建 Add 区域', ['add_region_overlay.png', 'add_region_mask.png', 'add_region_with_deleted_scene.png']),
    ('原生二维生成与 rembg', ['generated_2d_image.png', 'generated_2d_mask.png', 'generated_2d_rgba.png']),
    ('原生三维生成、优化和放置', ['generated_object_3d_preview.png', 'scene_inserted_before_refine.png',
                             'scene_final.png', 'scene_final_crop.png', 'final_comparison.png', 'final_multiview.png'])]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-name', default='native_all_cases_review.zip')
    args = parser.parse_args()
    assert Path(args.archive_name).name == args.archive_name and args.archive_name.endswith('.zip')
    summaries = []
    for case in ['case00', 'case02', 'case04', 'case07', 'case10']:
        folder = OUT / case
        if not (folder / 'config.json').exists():
            summaries.append(dict(case=case, status='not_run'))
            continue
        cfg = read(folder / 'config.json')
        prompt = read(folder / 'prompts.json')
        status = read(folder / 'records/status.json', {})
        localized = read(folder / 'records/localization.json', {})
        add = read(folder / 'records/add_region.json', {})
        quality = read(folder / 'records/visual_review.json', {})
        summary = dict(case=case, A=prompt['A'], B=prompt['B'], status=status.get('status', 'prepared'),
                       localization=localized, add_region=add, quality=quality,
                       execution_success=status.get('status') == 'complete',
                       output_path=str(folder), final_ply=str(folder / 'final_scene.ply'),
                       stages=status.get('stages', {}), post_insertion_refinement=False)
        summaries.append(summary)
        panels = []
        for title, filename in [('Original', 'original_view.png'), ('GE localization', 'localization_mask_overlay.png'),
                                ('Delete', 'delete_after.png'), ('GE-derived Add box', 'add_region_with_deleted_scene.png'),
                                ('Generated 2D', 'generated_2d_image.png'), ('Final 3D', 'scene_final.png')]:
            if (folder / filename).exists():
                panels.append((title, Image.open(folder / filename).convert('RGB')))
        if panels:
            canvas = Image.new('RGB', (3 * 512, 2 * 542), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (title, im) in enumerate(panels):
                x, y = i % 3 * 512, i // 3 * 542
                draw.text((x + 10, y + 8), title, fill='black')
                canvas.paste(ImageOps.contain(im, (512, 512)), (x, y + 30))
            canvas.save(folder / 'stages_overview.png')
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GaussianEditor 原生流程检查</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:24px auto;padding:20px;background:#f6f7f9}img{max-width:100%;height:auto}section{background:white;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}code{overflow-wrap:anywhere}figure{margin:0}figcaption{font-size:14px}a{color:#1261a6}.note{background:#fff3cc;padding:14px}</style>'''
        page += '<h1>' + case + '：GaussianEditor 原生定位 → Delete → Add</h1>'
        page += '<p>运行状态：' + html.escape(summary['status']) + '</p><p>A：' + html.escape(prompt['A']) + '</p><p>B：' + html.escape(prompt['B']) + '</p>'
        page += '<p>使用完整原始 COLMAP 相机集合。LangSAM 原生候选选择和 semantic tracing 得到旧设备高斯，Add 框来自这些高斯的实际可见 Mask。未读取 Ours 的目标框、参考视角或生成资产。</p>'
        page += '<p class="note">原生 Refine Text 用于插入前的 mesh-to-Gaussian/IP2P 优化。Add 插入后没有额外场景精细化：scene_inserted_before_refine.png 为兼容要求保留的名称，其内容是真实初次插入结果，与最终场景对应同一阶段。</p>'
        if quality:
            page += '<p class="note">实际检查：' + html.escape(quality.get('observations', str(quality))) + '</p>'
        page += '<p><a href="config.json">配置</a> · <a href="prompts.json">Prompt</a> · <a href="records/localization.json">定位记录</a> · <a href="records/add_region.json">Add 框来源</a> · <a href="run.log">完整运行日志</a></p>'
        page += '<p>最终 PLY 保存在服务器：<code>' + html.escape(summary['final_ply']) + '</code></p>'
        for title, names in STAGES:
            page += '<section><h2>' + title + '</h2><div class="grid">'
            for name in names:
                if (folder / name).exists():
                    page += '<figure><a href="' + name + '"><img src="' + name + '" loading="lazy"></a><figcaption>' + name + '</figcaption></figure>'
                else:
                    page += '<p>尚未输出：' + name + '</p>'
            page += '</div></section>'
        (folder / 'index.html').write_text(page + '</html>', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(dict(updated_at=time.time(), cases=summaries), indent=2, ensure_ascii=False))
    cards = []
    overview_rows = []
    for item in summaries:
        case = item['case']
        folder = OUT / case
        if not (folder / 'index.html').exists():
            continue
        cards.append('<section><h2><a href="' + case + '/index.html">' + case + '</a></h2><p>' +
                     html.escape(item['status']) + '</p><p>' + html.escape(item.get('quality', {}).get('observations', '')) +
                     '</p><a href="' + case + '/index.html"><img src="' + case + '/stages_overview.png"></a></section>')
        row = []
        for name in ['original_view.png', 'localization_mask_overlay.png', 'delete_after.png',
                     'generated_2d_image.png', 'scene_final.png']:
            if (folder / name).exists():
                row.append(Image.open(folder / name).convert('RGB'))
        if len(row) == 5:
            overview_rows.append((case, row))
    if overview_rows:
        overview = Image.new('RGB', (5 * 384, len(overview_rows) * 420), 'white')
        draw = ImageDraw.Draw(overview)
        labels = ['Original', 'GE localization', 'Native Delete', 'Generated 2D', 'Final 3D']
        for i, (case, row) in enumerate(overview_rows):
            for j, image in enumerate(row):
                draw.text((j * 384 + 8, i * 420 + 8), case + ' | ' + labels[j], fill='black')
                overview.paste(ImageOps.contain(image, (384, 384)), (j * 384, i * 420 + 28))
        overview.save(OUT / 'all_cases_overview.png')
    complete = sum(x.get('execution_success', False) for x in summaries)
    index = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GaussianEditor 原生流程五案例</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:24px auto;padding:20px;background:#f5f6f8}img{max-width:100%;height:auto}section{background:white;margin:20px 0;padding:20px}a{color:#165ca4}</style>'
    index += '<h1>GaussianEditor 原生流程：五案例</h1><p>运行完成 ' + str(complete) + '/5。运行完成不代表成功替换；各阶段的实际缺陷保留在图片和检查记录中。</p>'
    index += '<p>统一原生流程和参数：24 视角语义追踪、0.7 分割阈值、Delete 1500 步、NeuS 3000 步、mesh-to-Gaussian 原生 0–1800 循环、DPT Depth Scale=1.0。未使用 Ours 空间输入，也未调用 Qwen 或 TRELLIS。</p>'
    index += '<p><a href="summary.json">运行摘要与服务器 PLY 路径</a> · <a href="all_cases_overview.png">五案例总览</a></p>'
    index += ''.join(cards) + '</html>'
    (OUT / 'index.html').write_text(index, encoding='utf-8')
    archive = OUT / args.archive_name
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / 'summary.json', 'summary.json')
        z.write(OUT / 'index.html', 'index.html')
        if (OUT / 'all_cases_overview.png').exists():
            z.write(OUT / 'all_cases_overview.png', 'all_cases_overview.png')
        for item in summaries:
            folder = OUT / item['case']
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in ['.png', '.jpg', '.json', '.html', '.log', '.patch']:
                    if 'add' not in path.relative_to(folder).parts:
                        z.write(path, str(path.relative_to(OUT)))
        for path in sorted((BASE / 'native_gaussianeditor').glob('*')):
            if path.suffix in ['.py', '.md']:
                z.write(path, 'code/' + path.name)
    print('Inspection package:', archive, archive.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
