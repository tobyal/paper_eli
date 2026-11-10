"""Package one completed case and its actual configured refinement weight."""
import argparse
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args()
root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison/gsle_weak_disds')
folder=root/a.case
assert folder.resolve().parent==root.resolve()
status=json.loads((folder/'records/status.json').read_text())
assert status['status']=='complete'
v=json.loads((folder/'records/verification.json').read_text())
records=[json.loads(x) for x in (folder/'refine/training.jsonl').read_text().splitlines()]
assert [r['step'] for r in records]==list(range(1,v['iterations']+1))
assert all(r['densified']==0 and r['pruned']==0 for r in records)
assert all(all(r['gradient_norms'][k]==0 for k in ('xyz','log_scale','rotation','opacity_logit')) for r in records)
summary=dict(case=a.case,iterations=v['iterations'],sds_weight=v['sds_weight'],color_lr=v['color_lr'],
    geometry_unchanged=all(v['frozen_parameters_bitwise_equal'].values()),
    alpha_depth_unchanged=v['geometry_checks_passed'],background_unchanged=v['background_bitwise_identical'],
    source_preservation=json.loads((folder/'records/source_preservation.json').read_text()),
    rgb_mse=v['anchor_rgb_mse'],training_seconds=sum(r['seconds'] for r in records),outputs=v['outputs'])
(root/'case00_current_summary.json').write_text(json.dumps(summary,indent=2))
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{a.case} DI-SDS {v['sds_weight']}</title><style>body{{font:17px/1.6 system-ui;max-width:1550px;margin:auto;padding:24px;background:#f4f6f8}}img{{max-width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>{a.case} · 颜色参数 DI-SDS · SDS 权重 {v['sds_weight']}</h1><p>本次仅重新训练 {a.case}，共 {v['iterations']} 步。从原始 Coarse 6500 检查点开始，固定 xyz、scale、rotation、opacity，仅优化颜色，学习率 {v['color_lr']}。保留原有 DI-SDS 模型、深度条件、Inpainting Mask 和 Anchor RGB/Mask 监督。</p><p><a href="{a.case}/index.html">查看 100 / 500 / 1000 步、多视角和场景结果</a> · <a href="case00_current_summary.json">运行指标与 PLY 路径</a></p><a href="{a.case}/comparison.png"><img src="{a.case}/comparison.png"></a><a href="{a.case}/scene_comparison.png"><img src="{a.case}/scene_comparison.png"></a></html>'''
(root/'case00_current.html').write_text(page,encoding='utf-8')
archive=root/f'{a.case}_sds{v["sds_weight"]}_review.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['case00_current.html','case00_current_summary.json']:z.write(root/name,name)
    paths=[x for x in folder.iterdir() if x.is_file() and x.suffix in ('.png','.html','.json')]
    for sub in ['records','refine/previews']:
        paths.extend(x for x in (folder/sub).rglob('*') if x.is_file() and x.suffix in ('.png','.json','.patch'))
    for x in paths:z.write(x,x.relative_to(root))
print(json.dumps(summary,indent=2))
print(archive,archive.stat().st_size)
