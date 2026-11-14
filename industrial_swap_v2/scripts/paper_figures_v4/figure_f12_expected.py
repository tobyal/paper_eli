"""Explicit hypothetical F12 panel (b); preserves the real experimental figure."""
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
PREVIOUS = OUT / 'F12_multicase_training'
EMPIRICAL = OUT / 'F12_multicase_training_final'
DEST = OUT / 'F12_expected_illustration'


def main():
    provenance = read(OLD / 'sources.json')
    records = {r['case']: r for r in provenance['cases']}
    old_hashes = {str(p): sha(p) for p in OLD.rglob('*') if p.is_file()}
    old_hashes.update({str(p): sha(p) for p in PREVIOUS.rglob('*') if p.is_file()})
    old_hashes.update({str(p): sha(p) for p in EMPIRICAL.rglob('*') if p.is_file()})
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
        assert baseline['train'] == summary['baseline']['train']
        history = {0: {split: baseline[split] for split in ('train', 'validation')}}
        for ev in evaluations:
            step = int(ev['step'])
            metrics = {split: ev[split] for split in ('train', 'validation')}
            if step in history:
                assert metrics == history[step], f'{case}: duplicate step differs'
            history[step] = metrics
        assert history[summary['best_step']]['validation'] == summary['best']['validation']
        for key in ('rgb_l1', 'depth_mae'):
            assert np.isfinite(history[0]['validation'][key]) and history[0]['validation'][key] > 0
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
    def normalized(split, key):
        values = np.array([[h[t][split][key]/h[0]['validation'][key] for t in common] for h in histories])
        assert np.isfinite(values).all()
        return values
    rgb, depth = normalized('validation', 'rgb_l1'), normalized('validation', 'depth_mae')
    train_rgb, train_depth = normalized('train', 'rgb_l1'), normalized('train', 'depth_mae')
    # All five cases contribute equally at every actual shared step. No interpolation or smoothing.
    mean_rgb, sd_rgb = rgb.mean(0), rgb.std(0, ddof=1)
    mean_depth, sd_depth = depth.mean(0), depth.std(0, ddof=1)
    mean_train_rgb, mean_train_depth = train_rgb.mean(0), train_depth.mean(0)
    rows = []
    for i, (case, device, history, summary) in enumerate(zip(ORDER, DEVICES, histories, summaries)):
        for j, step in enumerate(common):
            ev, tr = history[step]['validation'], history[step]['train']
            rows.append(dict(record_type='case', case=case, device=device, step=step,
                training_rgb_l1=tr['rgb_l1'], training_depth_mae=tr['depth_mae'],
                validation_rgb_l1=ev['rgb_l1'], validation_depth_mae=ev['depth_mae'],
                relative_training_rgb_error=train_rgb[i, j], relative_training_depth_error=train_depth[i, j],
                relative_heldout_rgb_error=rgb[i, j], relative_heldout_depth_error=depth[i, j],
                foreground_psnr_db=ev['foreground_psnr'], normalized_depth_mae_percent=ev['depth_mae_relative']*100,
                normalization_validation_rgb_l1_step0=history[0]['validation']['rgb_l1'],
                normalization_validation_depth_mae_step0=history[0]['validation']['depth_mae'],
                selected_checkpoint_step=summary['best_step'], is_selected_checkpoint=step == summary['best_step'],
                metric_source='baseline_metrics.json' if step == 0 else 'evaluations.json'))
    for j, step in enumerate(common):
        rows.append(dict(record_type='aggregate', case='five_case_mean', device='All five devices', step=step,
            training_mean_relative_rgb_error=mean_train_rgb[j], training_mean_relative_depth_error=mean_train_depth[j],
            heldout_mean_relative_rgb_error=mean_rgb[j], heldout_std_relative_rgb_error=sd_rgb[j],
            heldout_mean_relative_depth_error=mean_depth[j], heldout_std_relative_depth_error=sd_depth[j], n_cases=5, std_ddof=1))

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
    assert lower_data == read(PREVIOUS/'sources.json')['lower_panel_data']

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'panels').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.45))
    fig.subplots_adjust(left=.095, right=.965, bottom=.14, top=.945, wspace=.32, hspace=.43)
    plt.setp(axes.ravel(), facecolor='white')
    for ax, training_mean, mean, sd, ylabel, title in (
        (axes[0,0], mean_train_rgb, mean_rgb, sd_rgb, 'Relative RGB Reconstruction Error', '(a) RGB Reconstruction Convergence'),
        (axes[0,1], mean_depth, mean_train_depth, sd_depth, 'Relative Depth Reconstruction Error', '(b) Depth Reconstruction Convergence')):
        ax.fill_between(common, mean-sd, mean+sd, color=COLORS[0], alpha=.12, linewidth=0, zorder=1)
        ax.plot(common, training_mean, color='#71869A', ls='--', lw=1.7, zorder=2)
        ax.plot(common, mean, color=COLORS[0], lw=2.0, marker='o', ms=2.4, zorder=3)
        assert max(np.max(training_mean), np.max(mean+sd)) < 1.25
        ax.set(xlabel='Training Iteration', ylabel=ylabel, title=title, xlim=(0, common[-1]), ylim=(0, 1.25))
        ax.set_xticks(np.arange(0, common[-1]+1, 1000))
        ax.set_yticks(np.arange(0, 1.26, .25))
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(handles=[Line2D([], [], color='#71869A', ls='--', lw=1.7, label='Training Mean'),
                           Line2D([], [], color=COLORS[0], lw=2, label='Held-out Mean'),
                           Patch(facecolor=COLORS[0], alpha=.12, edgecolor='none', label='Held-out ±1 SD')],
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
        'Expected-trend illustration, not a new experimental result. Only panel (b) is hypothetical: '
        'the two depth mean series are exchanged as requested for an expected-result mock-up. '
        'The illustrative Training Mean uses the original measured held-out mean, and the illustrative Held-out Mean '
        'uses the original measured training mean. The original held-out SD band width is retained and recentered on '
        'the illustrative held-out mean; it is not an observed SD for a newly measured held-out series. '
        'Panels (a), (c), and (d) remain identical to the empirical figure. No experiments, raw logs, or empirical '
        'figures were changed. The verified experimental figure remains in F12_multicase_training_final. '
        'This version is for conceptual discussion of expected behavior, not evidence of measured training/validation performance.')
    metadata = dict(artifact_type='expected_trend_illustration', modified_panel='b', hypothetical=True,
        empirical_figure=source(EMPIRICAL/'figure_F12_final.png'), empirical_sources=source(EMPIRICAL/'sources.json'),
        underlying_sources=sources, steps=common,
        illustrative_training_mean_source='original measured held-out depth mean',
        illustrative_heldout_mean_source='original measured training depth mean',
        illustrative_band='original held-out SD width recentered on illustrative held-out mean; not a measured held-out SD',
        unchanged_empirical_panels=['a','c','d'], no_new_training=True, original_inputs_and_f12_sha256_unchanged=True)
    fig.canvas.draw()
    fig.savefig(DEST/'figure_F12_expected.pdf', facecolor='white')
    fig.savefig(DEST/'figure_F12_expected.png', dpi=300, facecolor='white')
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(axes.ravel()):
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.04)
        fig.savefig(DEST/'panels'/f'{chr(97+i)}.png', dpi=300, bbox_inches=box, facecolor='white')
    plt.close(fig)
    for panel in ('a', 'c', 'd'):
        assert np.array_equal(np.asarray(Image.open(DEST/'panels'/f'{panel}.png')),
                              np.asarray(Image.open(EMPIRICAL/'panels'/f'{panel}.png'))), f'Unmodified panel {panel} changed'
    metadata['a_c_d_pixels_identical_to_empirical'] = True
    illustration_rows = [dict(artifact_type='hypothetical_illustration_not_measurement', step=int(t),
        illustrative_training_mean=float(mean_depth[j]), illustrative_heldout_mean=float(mean_train_depth[j]),
        illustrative_band_half_width=float(sd_depth[j])) for j,t in enumerate(common)]
    csvout(DEST/'illustrative_depth_curves.csv', illustration_rows)
    (DEST/'caption.txt').write_text(caption+'\n')
    for p, expected in protected.items():
        assert sha(p) == expected, f'Read-only input modified: {p}'
    save(DEST/'sources.json', metadata)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1800', '-png',
                    str(DEST/'figure_F12_expected.pdf'), str(DEST/'pdf_preview')], check=True)
    Image.open(DEST/'pdf_preview.png').convert('RGB').save(DEST/'preview_F12_expected.jpg', quality=95)
    print(json.dumps({'completed': str(DEST), 'artifact_type': 'hypothetical_expected_illustration',
                      'a_c_d_pixels_unchanged': True, 'empirical_files_unchanged': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
