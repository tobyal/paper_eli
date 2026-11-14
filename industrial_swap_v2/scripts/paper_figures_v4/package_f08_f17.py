"""Review pages, reproduction notes and provenance for the two final figures."""
import html, json, os, shutil, subprocess
from pathlib import Path

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
CSS='body{font:16px/1.6 system-ui;max-width:1450px;margin:auto;padding:28px;color:#222}img{max-width:100%}a{color:#155da1}table{border-collapse:collapse}td,th{padding:7px 20px;border-bottom:1px solid #ddd;text-align:right}figure{margin:15px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.note{background:#f5f7f9;padding:14px 18px}'

def main():
    f8=BASE/'F08_localization_final';f17=BASE/'F17_localization_comparison'
    for d,scripts in [(f8,['figure_f08_final.py','f08_instance_audit.json']),
        (f17,['evaluate_f17.py','figure_f17.py','f17_native_localize.py'])]:
        for s in scripts:shutil.copy2(Path(__file__).with_name(s),d/'code'/s)
        shutil.copy2(__file__,d/'code'/Path(__file__).name)
    text8='''F08: Reference-image-guided instance localization

Actual saved Case10 Gaussian projections, unchanged cameras and selections.
Ten confirmed M1 tasks contain nine physical targets: M1-04/M1-09 identify the
same cooler; archived case 11 is excluded. Preliminary AI visual audit gives
text-only 3/9 and photo-guided 9/9. This is not independent human adjudication.
Some saved semantic descriptions are imperfect; see caption and audit notes.
No simulated IoU table, generation, training or M2-M4 was used.

Render/reproduce on the server:
export ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli
cd "$ROOT"
envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f08_final.py

Figure: 180 x 63 mm; 300 DPI PNG; PDF text is embedded vector TrueType.
instance_selection.csv: one row per distinct physical target, all underlying M1 IDs.
sources.json records the unchanged image sources, camera and hashes.
'''
    (f8/'README.txt').write_text(text8)
    caption8=html.escape((f8/'caption.txt').read_text())
    (f8/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>F08 - Photo-guided localization</title><style>{CSS}</style>
<h1>F08 · 照片引导实例定位</h1><p>9 个去重物理目标；Text-only 3/9，Photo-guided 9/9。沿用实际定位结果，仍为 AI 初步核验。</p>
<p><a href="figure_F08.png">300 DPI PNG</a> · <a href="figure_F08.pdf">PDF</a> · <a href="instance_selection.csv">逐目标 CSV</a> · <a href="caption.txt">英文图注</a> · <a href="sources.json">来源与相机</a> · <a href="README.txt">复现说明</a> · <a href="../F17_localization_comparison/index.html">F17</a></p>
<img src="figure_F08.png" alt="F08"><p>{caption8}</p>''',encoding='utf-8')
    metrics=json.loads((f17/'metrics.json').read_text());views=json.loads((f17/'evaluation_views.json').read_text())['views']
    lines=['F17: GaussianEditor / DGE / Ours actual localization comparison','',
        'PRIMARY: six held-out camera-target pairs, four physical targets (M1-03/04/05/06).',
        'Two exclusions were defined independently of IoU:',
        '- M1-06_frame_000202: AI draft identifies another physical cooler.',
        '- M1-03_frame_000249: used by both baseline native tracing samples.',
        'All eight nonempty supplied drafts are retained unchanged and scored.',
        'All-draft sensitivity results are separate fields, not mixed into primary means.',
        'Twelve empty masks are missing data, not zero-IoU observations.',
        'Nine physical targets are indexed; five have no eligible GT and remain NA.',
        'Annotations are unreviewed AI polygons, not human-validated ground truth.',
        'Primary test cameras were unused by per-target native tracing/photo assistance;',
        'the underlying images still belong to the scene/language-field training capture.',
        '', 'Actual target-macro means:']
    for r in metrics['summary']:lines.append(f"{r['method']}: primary {r['mean_heldout_iou']:.6f}; all 8 draft sensitivity {r['all_8_drafts_sensitivity_mean']:.6f}")
    lines+=['','METHODS:',
        'GE: unchanged native first LangSAM candidate + Gaussian semantic tracing, 24 views / 0.7 threshold.',
        'DGE: unchanged native first LangSAM candidate + Gaussian semantic tracing, 20 views / 0.8 threshold.',
        'Two missing text queries were run for each method; existing native case04 was reused.',
        'M1-03 and M1-06 use identical text; baselines therefore share one prediction for those distinct targets.',
        'Ours: previously confirmed photo-assisted selections, no relocalization or retraining.',
        'Every method is evaluated on the same original 3DGS, 938x528 cameras and contribution threshold 0.05.',
        'GE/DGE have text-only input; Ours additionally has the registered reference-photo instance cue.',
        'Visualizations use actual registered photographs, not Gaussian RGB as the background.',
        'Shared target zooms include full-photo insets so off-target selected regions remain visible.',
        'All IoUs are full-frame; mean views per target, then unweighted mean targets.',
        '', 'REPRODUCE:', 'export ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli','cd "$ROOT"',
        '# Saved indices already exist. Evaluation/rendering only:',
        'envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/evaluate_f17.py --gpu 2',
        'envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f17.py',
        '# If a missing native localization needs reproducing, in an isolated output directory:',
        'baseline_comparison/envs/gaussianeditor_add/bin/python industrial_swap_v2/scripts/paper_figures_v4/f17_native_localize.py GaussianEditor 03 --gpu 2',
        'baseline_comparison/envs/gaussianeditor_add/bin/python industrial_swap_v2/scripts/paper_figures_v4/f17_native_localize.py GaussianEditor 05 --gpu 2',
        'envs/dge/bin/python industrial_swap_v2/scripts/paper_figures_v4/f17_native_localize.py DGE 03 --gpu 2',
        'envs/dge/bin/python industrial_swap_v2/scripts/paper_figures_v4/f17_native_localize.py DGE 05 --gpu 2',
        '', 'Native harness exits after localization. Completed jobs are reused, not overwritten.',
        'No diffusion guidance, Qwen, TRELLIS, editing, or optimizer step is executed.',
        'All provided scores in the earlier simulated 20-case table are explicitly unused.',
        'Figure: 180 x 133 mm; 300 DPI PNG; embedded vector PDF text.',
        'CSV blank values are NA. supplied_draft_iou includes excluded drafts for transparency.',
        'selection_indices/ stores each real saved selection; native_localization/ stores traces/candidates.',
        'evaluation_views.json records all exclusion reasons, camera matrices and annotation metadata.',
        'validation.json records independent PNG/CSV checks and unchanged original scene SHA256.']
    (f17/'README.txt').write_text('\n'.join(lines)+'\n')
    sources=json.loads((f17/'sources.json').read_text())
    sources['native_commits']={
        'GaussianEditor':json.loads((ROOT/'baseline_comparison/gaussianeditor_native/case04/config.json').read_text())['upstream_commit'],
        'DGE':json.loads((ROOT/'baseline_comparison/dge_native/case04/config.json').read_text())['upstream_commit']}
    (f17/'sources.json').write_text(json.dumps(sources,indent=2))
    summary=''.join(f"<tr><td>{r['method']}</td><td>{r['mean_heldout_iou']:.4f}</td><td>{r['all_8_drafts_sensitivity_mean']:.4f}</td></tr>" for r in metrics['summary'])
    rowtable=''
    for label in metrics['common_targets']:
        data=[next(r for r in metrics['per_target'] if r['label']==label and r['method']==m) for m in ['GaussianEditor','DGE','Ours']]
        rowtable+='<tr><td>'+label+'</td><td>'+str(data[0]['valid_views'])+'</td>'+''.join(f"<td>{r['heldout_iou']:.4f}</td>" for r in data)+'</tr>'
    details=''
    for v in views:
        if not v['ai_draft_nonempty']:continue
        tid=v['task_id'];status='主统计采用' if v['primary_included'] else '主统计排除：'+', '.join(v['exclusion_reasons'])
        details+=f'<details><summary>{tid} · {html.escape(status)}</summary><div class="grid">'
        for method in ['GT','GaussianEditor','DGE','Ours']:
            details+=f'<figure><figcaption>{method}</figcaption><a href="panels/{tid}_{method}.png"><img src="panels/{tid}_{method}.png"></a></figure>'
        details+='</div></details>'
    caption17=html.escape((f17/'caption.txt').read_text())
    (f17/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>F17 - Actual localization evaluation</title><style>{CSS}</style>
<h1>F17 · 三方法定位投影与 IoU</h1>
<p class="note">实际主统计：4 个物理目标、6 个合格视角。GT 仍为未经人工复核的 AI 草稿。另有 5 个目标缺少合格标注，记为 NA。所有结果来自实际 Gaussian 索引投影，没有使用模拟的 20 行 IoU 表。</p>
<p><a href="figure_F17.png">300 DPI PNG</a> · <a href="figure_F17.pdf">PDF</a> · <a href="iou_per_view.csv">逐视角 CSV</a> · <a href="iou_per_target.csv">逐目标 CSV</a> · <a href="iou_summary.csv">汇总 CSV</a> · <a href="caption.txt">英文图注</a> · <a href="README.txt">协议与复现</a> · <a href="../F08_localization_final/index.html">F08</a></p>
<img src="figure_F17.png" alt="F17"><h2>主结果及全部 8 张草稿的敏感性结果</h2><table><tr><th>方法</th><th>主统计：6 视角 / 4 目标</th><th>全部 8 草稿 / 4 目标</th></tr>{summary}</table>
<p>主统计排除 M1-06/frame202（草稿选错物理设备）及 M1-03/frame249（参与过两个基线的原生定位）。排除规则不依据 IoU；原 Mask 和全部分数均保留。</p>
<table><tr><th>目标</th><th>有效视角</th><th>GaussianEditor</th><th>DGE</th><th>Ours</th></tr>{rowtable}</table>
<h2>完整画面核查</h2><p>展开可检查所有视角的完整照片、真实预测和 GT，包括主图裁剪范围外的误选区域。</p>{details}<h2>Figure Caption</h2><p>{caption17}</p>''',encoding='utf-8')
    print('Review pages and reproduction notes written.')

if __name__=='__main__':main()
