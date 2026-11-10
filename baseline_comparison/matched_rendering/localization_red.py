"""Render exact native localization indices in red, without changing source PLYs.

The source is the PRE-EDIT scene: replaced/deleted target Gaussians need not exist
in final scenes. All selected indices, including disconnected false positives,
are retained. Cameras are copied exactly from the existing matched-view report.
"""
import argparse
import html
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import render as r

CASES = ['case00', 'case02', 'case04', 'case07', 'case10']
OUT = r.BASE / 'matched_views_localization_red'
METHODS = ['gaussianeditor', 'dge', 'gsle', 'ours']
LABELS = ['GaussianEditor', 'DGE', 'GSLE', 'Ours']
NOTE = 'Red = all actually selected pre-edit Gaussians; native opacity and occlusion preserved.'


def board(path, rows, title, subtitle=NOTE, cell=500):
    canvas = Image.new('RGB', (cell * len(rows[0]), 80 + len(rows) * (cell + 40)), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((16, 10), title, font=ImageFont.truetype('DejaVuSans.ttf', 27), fill='black')
    d.text((16, 47), subtitle, font=ImageFont.truetype('DejaVuSans.ttf', 20), fill='#444444')
    for i, row in enumerate(rows):
        for j, (label, file, crop) in enumerate(row):
            x, y = j * cell, 80 + i * (cell + 40)
            d.text((x + 10, y + 7), label, font=ImageFont.truetype('DejaVuSans.ttf', 24), fill='black')
            im = Image.open(file).convert('RGB')
            if crop:
                im = im.crop(crop)
            im = ImageOps.contain(im, (cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + (cell-im.width)//2, y + 40 + (cell-im.height)//2))
    canvas.save(path, dpi=(300, 300))


def index_path(row, method):
    if method == 'ours':
        return Path(row['target_record']).parent / 'target_indices.npy'
    return r.BASE / (method + '_native') / row['case'] / (
        'localization/selected_indices.npy' if method == 'gaussianeditor' else 'target_indices.npy')


@torch.no_grad()
def run_case(row, scene):
    started = time.time()
    case = row['case']
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'localization_report.json').exists(), 'Preserve completed output'
    camera_path = r.BASE / 'matched_views' / case / 'camera_params.json'
    params = r.read(camera_path)
    common = r.c.Camera.from_record(params['reference_camera_resized'])
    r.c.save_image(out / 'original.png', r.c.render(common, scene, aux=False)['rgb'])
    report = dict(case=case, visualization='Native localization in red on the pre-edit scene',
                  camera_source=str(camera_path), camera_sha256=r.sha(camera_path),
                  training_steps=0, modified_ply_files=0, color_override=[1, 0, 0],
                  full_selection_used=True, geometry_and_opacity_unchanged=True,
                  note=NOTE, methods={}, crops={}, input_hashes={})
    r.save(out / 'camera_params.json', params)
    for method in METHODS:
        path = index_path(row, method)
        before = r.sha(path)
        ids = np.load(path, allow_pickle=False)
        assert ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
        assert len(ids) and ids.min() >= 0 and ids.max() < len(scene.xyz)
        assert len(np.unique(ids)) == len(ids)
        idx = torch.as_tensor(ids, device='cuda')
        selected = torch.zeros_like(scene.xyz)
        selected[idx] = 1
        record = dict(indices_path=str(path), indices_sha256=before, selected_gaussians=len(ids),
                      all_disconnected_regions_retained=True, views={})
        views = [(v['view'], r.c.Camera.from_record(v['matched_camera']))
                 for v in params['methods'][method]['views']]
        views.append(('same_camera', common))
        for view, camera in views:
            folder = out / 'views' / view
            folder.mkdir(parents=True, exist_ok=True)
            # Render native background colors and red target colors together in
            # one rasterization, retaining actual depth sorting/transmittance.
            colors = scene.colors(camera.center)
            original = r.c.raster(camera, scene, colors)
            colors[idx] = colors.new_tensor([1., 0., 0.])
            red = r.c.raster(camera, scene, colors)
            contribution = r.c.raster(camera, scene, selected)[0]
            r.c.save_image(folder / (method + '.png'), red)
            r.c.save_image(folder / ('original_' + method + '.png'), original)
            r.c.save_image(folder / (method + '_visible_selection.png'), contribution[None])
            a = contribution.cpu().numpy()
            stats = r.alpha_stats(a)
            outside = contribution == 0
            unchanged_error = float((red-original).abs()[:, outside].max()) if outside.any() else 0.
            assert unchanged_error < 2e-6, unchanged_error
            record['views'][view] = dict(camera=camera.record(), visible_selection=stats,
                                         outside_contribution_max_rgb_error=unchanged_error)
            if view == 'main':
                shutil.copy2(folder / (method + '.png'), out / (method + '.png'))
                report['crops'][method] = r.square_crop(stats['bbox'], camera.width)
            print(case, method, view, stats['pixels'], flush=True)
        assert r.sha(path) == before
        report['methods'][method] = record
        report['input_hashes'][str(path)] = before
        del colors, selected, red, original, contribution
    report['crops']['original'] = report['crops']['ours']
    columns = ['original'] + METHODS
    labels = ['Original'] + LABELS
    board(out / 'comparison_main.png', [[(l, out/(m+'.png'), None) for m,l in zip(columns,labels)]],
          case+' | Localization - Viewpoint-Matched Rendering')
    board(out / 'comparison_crops.png', [[(l, out/(m+'.png'), report['crops'][m]) for m,l in zip(columns,labels)]],
          case+' | Visible localization crops')
    board(out / 'comparison_multiview.png', [[(l+f' / aux {i}', out/'views'/f'aux{i}'/(m+'.png'), None)
          for m,l in zip(METHODS,LABELS)] for i in (1,2)], case+' | Localization - auxiliary matched views')
    board(out / 'comparison_same_camera.png', [[('Original',out/'original.png',None)] +
          [(l, out/'views/same_camera'/(m+'.png'), None) for m,l in zip(METHODS,LABELS)]],
          case+' | Localization - same Ours anchor for all methods',
          'Same-camera diagnostic | Full native selection in red; targets may be outside this view.')
    board(out / 'localization_before_after.png', [[(l+' / Original',out/'views/main'/('original_'+m+'.png'),None),
          (l+' / Selected',out/(m+'.png'),None)] for m,l in zip(METHODS,LABELS)],
          case+' | Same-camera localization controls', 'Pre-edit scene | Red = selected Gaussians')
    report['seconds'] = time.time()-started
    report['input_files_unchanged'] = r.sha(camera_path) == report['camera_sha256']
    assert report['input_files_unchanged']
    r.save(out/'localization_report.json',report)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>{case} 定位高斯红色显示</title><style>body{{font:17px/1.6 system-ui;max-width:1700px;margin:auto;padding:24px}}img{{max-width:100%}}section{{margin:28px 0}}</style>
    <h1>{case}：各方法实际定位高斯</h1><p><a href="../index.html">全部案例</a> | <a href="localization_report.json">定位索引与可见贡献记录</a> | <a href="camera_params.json">相机参数</a></p>
    <p>在编辑前原始场景中，将各方法保存的全部定位高斯以纯红色渲染。保留原有高斯位置、尺度、旋转、透明度及真实遮挡；未选中高斯保持原色。不是三维包围盒投影，也不是新设备的轮廓。</p>
    <p>主图与上一版最终结果对比采用完全相同的各方法相机；GaussianEditor 的 Add 位置与其原始定位可能不同，因此红色区域可能偏离画面中心。DGE 显示全部定位索引，不限于上一版用于取景的连通区域。</p>'''
    for name,label in [('comparison_main','匹配视角定位对比'),('comparison_crops','局部放大'),
                       ('comparison_same_camera','统一 Ours 相机定位对比'),('comparison_multiview','两个辅助视角'),
                       ('localization_before_after','各方法原场景 / 红色定位')]:
        page += f'<section><h2>{label}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png"></a></section>'
    page += '<p>'+' · '.join(f'{label}: {report["methods"][m]["selected_gaussians"]:,} Gaussians' for m,label in zip(METHODS,LABELS))+'</p></html>'
    (out/'index.html').write_text(page,encoding='utf-8')


def package(cases):
    overview = Image.new('RGB',(2500,90+540*len(cases)),'white')
    same = overview.copy()
    for canvas,kind in [(overview,'Matched views'),(same,'Same Ours camera')]:
        d=ImageDraw.Draw(canvas)
        d.text((16,12),kind+' | Native localization in red',font=ImageFont.truetype('DejaVuSans.ttf',27),fill='black')
        d.text((16,50),NOTE,font=ImageFont.truetype('DejaVuSans.ttf',22),fill='#444444')
    for i,case in enumerate(cases):
        for canvas,name in [(overview,'comparison_main'),(same,'comparison_same_camera')]:
            im=Image.open(OUT/case/(name+'.png'))
            canvas.paste(im.crop((0,80,2500,620)),(0,90+i*540))
            ImageDraw.Draw(canvas).text((12,90+i*540+38),case,font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
    for canvas,name in [(overview,'overview_all_cases'),(same,'overview_same_camera')]:
        canvas.save(OUT/(name+'.png'),dpi=(300,300))
        ImageOps.contain(canvas,(1600,1900)).save(OUT/(name+'_web.jpg'),quality=94)
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>定位高斯红色对比</title><style>body{font:17px/1.7 system-ui;max-width:1600px;margin:auto;padding:24px}img{max-width:100%}</style><h1>五个案例：定位高斯红色显示</h1><p>红色表示各方法实际保存的全部定位高斯，包括 Ours M1；显示在编辑前原始场景中，保留真实透明度和遮挡。所有三维文件不变，无训练。</p><p>匹配视角与上一版完全一致；额外提供统一 Ours 相机的定位对照。远处误选区域和定位泄漏不被人为清除。</p><p>'''+ ' | '.join(f'<a href="{case}/index.html">{case}</a>' for case in cases)+'''</p><h2>匹配视角</h2><a href="overview_all_cases.png"><img src="overview_all_cases_web.jpg"></a><h2>统一 Ours 相机</h2><a href="overview_same_camera.png"><img src="overview_same_camera_web.jpg"></a></html>''',encoding='utf-8')
    (OUT/'figure_caption.txt').write_text('Native Gaussian localization visualized in red on the pre-edit source scene. All saved per-method target indices are used, including disconnected regions. Red colors are overridden only during Gaussian rasterization; native geometry, opacity, occlusion and nonselected appearance are preserved. Matched-view cameras are identical to the previously exported final-scene comparison. An additional same-camera diagnostic uses the Ours anchor for every method. These images visualize localization, not generated equipment. No PLY was modified.')
    with zipfile.ZipFile(OUT/'localization_red_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.jpg','.json','.html','.txt'):
                z.write(p,p.relative_to(OUT))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',default=CASES);args=parser.parse_args()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    OUT.mkdir(parents=True,exist_ok=True)
    spec=r.read(r.BASE/'prompts.json');before=r.sha(spec['source_scene'])
    scene=r.c.GaussianData.load(spec['source_scene'])
    for case in args.cases:
        run_case(next(x for x in spec['cases'] if x['case']==case),scene)
    assert before == r.sha(spec['source_scene'])
    r.save(OUT/'validation.json',dict(source_scene=spec['source_scene'],source_sha256=before,
        source_unchanged=True,cases=args.cases,training_steps=0,ply_writes=0,all_native_indices_used=True))
    package(args.cases)
    print('COMPLETE localization red',args.cases,flush=True)
