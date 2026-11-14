"""Revise F12 using five actual held-out histories; read-only experiment inputs."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from figure_common import OUT, COLORS, read, source, sha, save, csvout, plt

ORDER = ['case00', 'case02', 'case04', 'case07', 'case10']
DEVICES = ['Robot', 'Conveyor', 'Chiller', 'Storage Cabinet', 'Tool Cabinet']
OLD = OUT / 'F12_m3_training'
DEST = OUT / 'F12_multicase_training'


def main():
    provenance = read(OLD / 'sources.json')
    records = {r['case']: r for r in provenance['cases']}
    old_hashes = {str(p): sha(p) for p in OLD.rglob('*') if p.is_file()}
    protected = dict(old_hashes)
    histories, summaries, sources = [], [], []
    for case, device in zip(ORDER, DEVICES):
        record = records[case]
        ep, sp = Path(record['evaluations']['path']), Path(record['summary']['path'])
        assert sha(ep) == record['evaluations']['sha256'], f'{case}: original evaluations changed'
        assert sha(sp) == record['summary']['sha256'], f'{case}: original summary changed'
        baseline_path = sp.parent / 'baseline_metrics.json'
        summary, baseline, evaluations = read(sp), read(baseline_path), read(ep)
        assert baseline['validation'] == summary['baseline']['validation']
        history = {0: baseline['validation']}
        for ev in evaluations:
            step = int(ev['step'])
            if step in history:
                assert ev['validation'] == history[step], f'{case}: duplicate step differs'
            history[step] = ev['validation']
        assert history[summary['best_step']] == summary['best']['validation']
        for key in ('rgb_l1', 'depth_mae'):
            assert np.isfinite(history[0][key]) and history[0][key] > 0
        view_counts = {split: sum(v['split'] == split for v in baseline['views'])
                       for split in ('train', 'validation')}
        assert view_counts == {'train': 30, 'validation': 6}
        for p in (ep, sp, baseline_path):
            protected[str(p)] = sha(p)
        sources.append(dict(case=case, device=device, evaluations=source(ep), summary=source(sp),
                            baseline=source(baseline_path), actual_steps=sorted(history),
                            selected_checkpoint_step=summary['best_step'], view_counts=view_counts))
        histories.append(history)
        summaries.append(summary)

    common = sorted(set.intersection(*(set(h) for h in histories)))
    assert common[0] == 0 and len(common) > 1
    rgb = np.array([[h[t]['rgb_l1']/h[0]['rgb_l1'] for t in common] for h in histories])
    depth = np.array([[h[t]['depth_mae']/h[0]['depth_mae'] for t in common] for h in histories])
    assert np.isfinite(rgb).all() and np.isfinite(depth).all()
    # All five cases contribute equally at every actual shared step. No interpolation or smoothing.
    mean_rgb, sd_rgb = rgb.mean(0), rgb.std(0, ddof=1)
    mean_depth, sd_depth = depth.mean(0), depth.std(0, ddof=1)
    rows = []
    for i, (case, device, history, summary) in enumerate(zip(ORDER, DEVICES, histories, summaries)):
        for j, step in enumerate(common):
            ev = history[step]
            rows.append(dict(record_type='case', case=case, device=device, step=step,
                validation_rgb_l1=ev['rgb_l1'], validation_depth_mae=ev['depth_mae'],
                relative_rgb_error=rgb[i, j], relative_depth_error=depth[i, j],
                foreground_psnr_db=ev['foreground_psnr'], normalized_depth_mae_percent=ev['depth_mae_relative']*100,
                initial_rgb_l1=history[0]['rgb_l1'], initial_depth_mae=history[0]['depth_mae'],
                selected_checkpoint_step=summary['best_step'], is_selected_checkpoint=step == summary['best_step'],
                metric_source='baseline_metrics.json' if step == 0 else 'evaluations.json'))
    for j, step in enumerate(common):
        rows.append(dict(record_type='aggregate', case='five_case_mean', device='All five devices', step=step,
            mean_relative_rgb_error=mean_rgb[j], std_relative_rgb_error=sd_rgb[j],
            mean_relative_depth_error=mean_depth[j], std_relative_depth_error=sd_depth[j], n_cases=5, std_ddof=1))

    # Explicitly verify that every lower-row bar equals the original F12 CSV value.
    with (OLD/'data.csv').open(newline='') as f:
        original_rows = list(csv.DictReader(f))
    lower_data = []
    for case, device, summary in zip(ORDER, DEVICES, summaries):
        for stage, key in [('Initial', 'baseline'), ('PBR optimized', 'best')]:
            old = next(r for r in original_rows if r['record'] == 'summary' and r['case'] == case
                       and r['stage'] == stage and r['split'] == 'validation')
            v = summary[key]['validation']
            assert float(old['foreground_psnr']) == v['foreground_psnr']
            assert float(old['depth_mae_relative']) == v['depth_mae_relative']
            lower_data.append(dict(case=case, device=device, stage=stage,
                step=0 if key == 'baseline' else summary['best_step'],
                foreground_psnr_db=v['foreground_psnr'], normalized_depth_mae_percent=v['depth_mae_relative']*100))

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'panels').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.45))
    fig.subplots_adjust(left=.095, right=.965, bottom=.14, top=.945, wspace=.32, hspace=.43)
    plt.setp(axes.ravel(), facecolor='white')
    for ax, values, mean, sd, ylabel, title in (
        (axes[0,0], rgb, mean_rgb, sd_rgb, 'Relative Held-out RGB Error', '(a) RGB Reconstruction Convergence'),
        (axes[0,1], depth, mean_depth, sd_depth, 'Relative Held-out Depth Error', '(b) Depth Reconstruction Convergence')):
        for y in values:
            ax.plot(common, y, color='#7B8996', alpha=.32, lw=.65, zorder=1)
        ax.fill_between(common, mean-sd, mean+sd, color=COLORS[0], alpha=.17, linewidth=0, zorder=2)
        ax.plot(common, mean, color=COLORS[0], lw=2.0, marker='o', ms=2.4, zorder=3)
        ax.set(xlabel='Training Iteration', ylabel=ylabel, title=title, xlim=(0, common[-1]), ylim=(0, 1.06))
        ax.set_xticks(np.arange(0, common[-1]+1, 1000))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(handles=[Line2D([], [], color=COLORS[0], lw=2, label='Five-case mean'),
                           Patch(facecolor=COLORS[0], alpha=.17, edgecolor='none', label='±1 SD across cases'),
                           Line2D([], [], color='#7B8996', lw=.65, alpha=.5, label='Individual cases')],
                  frameon=False, loc='upper right', fontsize=7.7, handlelength=2.1, labelspacing=.35)

    x, width = np.arange(5), .35
    # The case order, paired bar geometry, colors and numerical data are unchanged.
    labels = ['00\nRobot', '02\nConveyor', '04\nChiller', '07\nStorage\nCabinet', '10\nTool\nCabinet']
    for ax, key, factor, ylabel, title in (
        (axes[1,0], 'foreground_psnr', 1, 'Foreground PSNR (dB)', '(c) Held-out RGB Reconstruction'),
        (axes[1,1], 'depth_mae_relative', 100, 'Normalized Depth MAE (%)', '(d) Held-out Depth Reconstruction')):
        for offset, metric_key, label, color in [(-.5, 'baseline', 'Initial', COLORS[4]), (.5, 'best', 'PBR optimized', COLORS[0])]:
            ax.bar(x+offset*width, [s[metric_key]['validation'][key]*factor for s in summaries],
                   width, label=label, color=color)
        ax.set_xticks(x, labels)
        ax.set(ylabel=ylabel, title=title)
        if key == 'foreground_psnr':
            ax.set_ylim(0, 32)
            ax.set_yticks(np.arange(0, 31, 5))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=9.3, pad=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)

    caption = (
        'Figure 12. Quantitative analysis of multi-view RGB-D Gaussian asset reconstruction across five industrial devices: '
        'Case00 (Robot), Case02 (Conveyor), Case04 (Chiller), Case07 (Storage Cabinet), and Case10 (Tool Cabinet). '
        '(a,b) At each common evaluation step, the held-out RGB L1 or absolute depth MAE of each case is divided by its own '
        'Step-0 value, then averaged with equal weight across the five cases. Thick solid curves show the mean; shaded regions '
        'show plus/minus one sample standard deviation (n=5, ddof=1); faint lines show individual histories. Markers indicate '
        'actual shared evaluations at Step 0 and every 250 iterations through Step 4000. Step-0 measurements are read from '
        'the existing baseline_metrics.json files because evaluations.json begins at Step 250. No interpolation, smoothing, '
        'best-so-far filtering, or new training is used. (c,d) The unchanged initial-versus-optimized comparisons use each '
        'asset\'s originally selected best validation checkpoint: 2250, 4000, 2000, 4000, and 4000 iterations, respectively. '
        'Foreground PSNR is reported in dB; normalized depth MAE is the absolute depth MAE divided by the local asset diagonal '
        'and multiplied by 100. All cases used 30 training and six held-out PBR views. These metrics measure reconstruction '
        'of the fixed GLB/PBR targets rather than independent real-world geometry or relighting accuracy.')
    metadata = dict(cases=sources, common_actual_steps=common, equal_case_weight=True,
        normalization='Validation RGB L1(t)/RGB L1(0); Validation absolute Depth MAE(t)/Depth MAE(0)',
        standard_deviation={'ddof': 1, 'n': 5, 'meaning': 'between-case sample SD, not confidence interval'},
        step0_source='baseline_metrics.json; verified equal to original F12 summary baseline',
        smoothing=False, interpolation=False, best_so_far_filter=False, no_new_training=True,
        lower_panel_data_unchanged=True, lower_panel_data=lower_data, original_f12=source(OLD/'sources.json'),
        last_common_step=common[-1], final_mean_relative_rgb=float(mean_rgb[-1]), final_mean_relative_depth=float(mean_depth[-1]),
        final_sample_sd_rgb=float(sd_rgb[-1]), final_sample_sd_depth=float(sd_depth[-1]),
        original_inputs_and_f12_sha256_unchanged=True)
    fig.canvas.draw()
    fig.savefig(DEST/'figure_F12_revised.pdf', facecolor='white')
    fig.savefig(DEST/'figure_F12_revised.png', dpi=300, facecolor='white')
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(axes.ravel()):
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.04)
        fig.savefig(DEST/'panels'/f'{chr(97+i)}.png', dpi=300, bbox_inches=box, facecolor='white')
    plt.close(fig)
    csvout(DEST/'multicase_convergence.csv', rows)
    (DEST/'caption.txt').write_text(caption+'\n')
    for p, expected in protected.items():
        assert sha(p) == expected, f'Read-only input modified: {p}'
    save(DEST/'sources.json', metadata)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1800', '-png',
                    str(DEST/'figure_F12_revised.pdf'), str(DEST/'pdf_preview')], check=True)
    Image.open(DEST/'pdf_preview.png').convert('RGB').save(DEST/'preview_F12_revised.jpg', quality=95)
    print(json.dumps({'completed': str(DEST), 'common_steps': common, 'csv_rows': len(rows),
                      'final_mean_relative_rgb': float(mean_rgb[-1]), 'final_mean_relative_depth': float(mean_depth[-1]),
                      'bottom_bars_unchanged': True, 'original_files_unchanged': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
