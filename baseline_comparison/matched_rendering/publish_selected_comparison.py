"""Package the accepted case00/02 revisions plus selected case04/07/10 updates.

No camera selection, rasterization, model modification or training is performed.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

SOURCES = {
    'case00': 'case00_ours_view_v2',
    'case02': 'case02_ours_view_v2',
    'case04': 'case04_ge_joint_view_v4',
    'case07': 'case07_gsle_ours_view_v2',
    'case10': 'case10_gsle_wider_view_v2',
}
NOTES = {
    'case00': '保留已确认的 00 更新版：四种方法共用 Ours 相机，红色定位为半透明。',
    'case02': '保留已确认的 02 更新版：四种方法共用 Ours 相机，红色定位为半透明。',
    'case04': 'GaussianEditor、DGE 使用确认的原展示视角；GE 同时包含分割和生成位置。GSLE 与 Ours 共用相机。',
    'case07': 'GaussianEditor、DGE 使用确认的原展示视角；GSLE 改为与 Ours 共用相机。',
    'case10': 'GaussianEditor、DGE 使用确认的原展示视角；GSLE 对齐 Ours 方向，围绕自身目标拉远，相机位置不同。',
}
FIGURES = [('comparison_paired', '定位与最终结果'), ('comparison_final', '最终结果'),
           ('comparison_localization', '定位结果'), ('comparison_crops', '局部放大'),
           ('comparison_aux1', '辅助视角 1'), ('comparison_aux2', '辅助视角 2')]
STYLE = '''<style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px;background:#f4f6f8;color:#18212b}img{max-width:100%;height:auto}section{margin:28px 0;background:white;padding:20px;border-radius:8px}nav{background:white;padding:12px;position:sticky;top:0}a{color:#155e9c}.note{padding:14px;background:#e7eff9;border-radius:6px}</style>'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header(title):
    return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title>{STYLE}'


def main(root):
    base = root/'baseline_comparison'
    out = base/'comparison_updated_v6'
    assert not out.exists(), 'Preserve completed package; choose a new revision.'
    out.mkdir(parents=True)
    manifest = dict(revision=out.name, cases={}, training_steps=0, ply_writes=0,
        rerendered=False, policy='Preserve accepted case00/02 exactly; incorporate selected case04/07/10 revisions in a separate standalone review package.')
    for case, revision in SOURCES.items():
        source = base/'target_centered_views'/revision
        target = out/case
        target.mkdir()
        hashes = {}
        for p in sorted(source.rglob('*')):
            relative = p.relative_to(source)
            # Only the deliverable views, paired figures and their source records.
            keep = (p.suffix == '.png' and (len(relative.parts) > 1 and relative.parts[0] == 'views'
                    or len(relative.parts) == 1 and p.name.startswith(('comparison_', 'original', 'ours_', 'gsle_', 'dge_', 'gaussianeditor_'))))
            keep |= len(relative.parts) == 1 and p.name in ('camera_params.json', 'report.json')
            if not p.is_file() or not keep:
                continue
            dest = target/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            hashes[str(relative)] = sha(p)
            assert sha(dest) == hashes[str(relative)]
        page = header(case+' · 五案例更新版')
        page += f'<h1>{case} · 五案例更新版</h1><p class="note">{NOTES[case]}</p><p><a href="../index.html">五案例更新版总览</a> | <a href="camera_params.json">相机记录</a> | <a href="report.json">来源与验证记录</a></p>'
        page += '<p>上排定位、下排最终结果，每列使用同一相机。不同方法之间是否共用相机按本案例说明；GSLE 为原始 Native DI-SDS 结果。</p>'
        for name, label in FIGURES:
            assert (target/(name+'.png')).exists()
            page += f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>'
        (target/'index.html').write_text(page+'</html>', encoding='utf-8')
        manifest['cases'][case] = dict(source_directory=str(source), source_revision=revision,
            explanation=NOTES[case], copied_sha256=hashes, copied_byte_identical=True)
    for filename, suffix in [('overview_localization_final', 'paired'), ('overview_final', 'final'),
                             ('overview_localization', 'localization')]:
        panels = [Image.open(out/case/f'comparison_{suffix}.png').convert('RGB') for case in SOURCES]
        assert len({p.width for p in panels}) == 1
        canvas = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
        y = 0
        for panel in panels:
            canvas.paste(panel, (0, y)); y += panel.height
        canvas.save(out/(filename+'.png'), dpi=(300, 300))
        ImageOps.contain(canvas, (1200, 4000)).save(out/(filename+'_web.jpg'), quality=94)
    page = header('五案例更新版 · 保留已确认的 00、02')
    page += '<h1>五案例更新版：00、02 保留，04、07、10 合并更新</h1><p class="note">本页以此前确认的 case00、case02 更新版为基础，合并最新三例；所有已选图片逐字节复制，没有再次选角、训练或修改三维场景。</p>'
    page += '<nav>'+' · '.join(f'<a href="#{c}">{c}</a>' for c in SOURCES)+'</nav>'
    page += '<p><a href="overview_localization_final.png">五案例高清总图</a> | <a href="overview_final.png">最终结果总图</a> | <a href="overview_localization.png">定位总图</a> | <a href="manifest.json">版本来源</a></p>'
    for case in SOURCES:
        page += f'<section id="{case}"><h2><a href="{case}/index.html">{case} · 完整结果</a></h2><p>{NOTES[case]}</p><a href="{case}/comparison_paired.png"><img src="{case}/comparison_paired.png" loading="lazy"></a></section>'
    page += '<p>所有方法的定位与最终结果在各自列内共用相机。跨方法可有不同相机位置，并非全部为同世界相机的像素级比较。GSLE 使用原始 Native DI-SDS；失败结果如实保留。</p></html>'
    (out/'index.html').write_text(page, encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    with zipfile.ZipFile(out/(out.name+'.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.png', '.jpg', '.json', '.html'):
                archive.write(p, p.relative_to(out))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli'))
    main(parser.parse_args().root)
