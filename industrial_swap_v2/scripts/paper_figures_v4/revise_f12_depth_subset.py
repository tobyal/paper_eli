"""F12(b): documented leave-Case04-out depth summary; preserve all raw records.

Run with system python3. Only panel b changes; no training or data fabrication.
"""
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_common import COLORS, plt, read, source, sha, save, csvout

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiment figure/F12'
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F12_depth_four_case_summary'
ORDER = ['case00', 'case02', 'case04', 'case07', 'case10']
EXCLUDE = 'case04'
INCLUDE = [x for x in ORDER if x != EXCLUDE]


def main():
    assert not (OUT / 'sources.json').exists(), 'Preserve a completed revision.'
    for folder in ('panels', 'data', 'inputs', 'code'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    prior = read(BASE / 'sources.json')
    before = {p.relative_to(BASE).as_posix(): sha(p) for p in BASE.rglob('*') if p.is_file()}
    for name in ('sources.json', 'caption.txt', 'delivery_manifest.json'):
        shutil.copy2(BASE / name, OUT / 'inputs' / ('previous_F12_' + name))
    shutil.copy2(BASE / 'multicase_convergence.csv', OUT / 'data/five_case_convergence_original.csv')
    rows = list(csv.DictReader((BASE / 'multicase_convergence.csv').open(encoding='utf-8-sig')))
    records = {(r['case'], int(r['step'])): r for r in rows if r['record_type'] == 'case'}
    steps = sorted({step for case, step in records})
    audit = []
    # Verify the original records; high Train/Held-out ratios alone are not invalid data.
    for case in prior['cases']:
        for key in ('evaluations', 'baseline', 'summary'):
            assert sha(case[key]['path']) == case[key]['sha256']
        history = [{**read(case['baseline']['path']), 'step': 0}] + read(case['evaluations']['path'])
        max_discrepancy = 0.
        for h in history:
            r = records[(case['case'], int(h['step']))]
            for split, csv_key in [('train', 'training_depth_mae'), ('validation', 'validation_depth_mae')]:
                values = np.array([v['depth_mae'] for v in h['views'] if v['split'] == split])
                assert len(values) == (30 if split == 'train' else 6)
                assert np.isfinite(values).all() and (values >= 0).all()
                error = abs(float(values.mean()) - h[split]['depth_mae'])
                max_discrepancy = max(max_discrepancy, error)
                assert error < 1e-12 and float(r[csv_key]) == h[split]['depth_mae']
        audit.append({'case': case['case'], 'original_hashes_match': True,
                      'maximum_per_view_aggregate_discrepancy': max_discrepancy,
                      'nonfinite_or_negative_metrics': False})

    train = np.array([[float(records[(case, t)]['relative_training_depth_error']) for t in steps] for case in INCLUDE])
    held = np.array([[float(records[(case, t)]['relative_heldout_depth_error']) for t in steps] for case in INCLUDE])
    mean_train, mean_held, sd_held = train.mean(0), held.mean(0), held.std(0, ddof=1)
    aggregates = {int(r['step']): r for r in rows if r['record_type'] == 'aggregate'}
    rgb_train = np.array([float(aggregates[t]['training_mean_relative_rgb_error']) for t in steps])
    rgb_mean = np.array([float(aggregates[t]['heldout_mean_relative_rgb_error']) for t in steps])
    rgb_sd = np.array([float(aggregates[t]['heldout_std_relative_rgb_error']) for t in steps])
    depth_rows = [dict(step=t, included_cases=';'.join(INCLUDE), excluded_case=EXCLUDE, n_cases=4,
                       training_mean_relative_depth_error=float(mean_train[i]),
                       heldout_mean_relative_depth_error=float(mean_held[i]),
                       heldout_std_relative_depth_error=float(sd_held[i]), std_ddof=1,
                       analysis='Post-hoc leave-Case04-out sensitivity summary; not invalid-record removal')
                  for i, t in enumerate(steps)]
    csvout(OUT / 'depth_convergence_four_cases.csv', depth_rows)
    # Keep every individual observation; separate the depth aggregation from panel a.
    updated_rows = []
    for r in rows:
        row = dict(r)
        if r['record_type'] == 'case':
            row['included_in_panel_b'] = r['case'] in INCLUDE
        else:
            i = steps.index(int(r['step']))
            row.update(case='panel_specific_mean', device='RGB: five devices; depth: four devices',
                       training_mean_relative_depth_error=float(mean_train[i]),
                       heldout_mean_relative_depth_error=float(mean_held[i]),
                       heldout_std_relative_depth_error=float(sd_held[i]),
                       n_cases='', n_cases_rgb=5, n_cases_depth=4, depth_excluded_case=EXCLUDE)
        updated_rows.append(row)
    csvout(OUT / 'multicase_convergence.csv', updated_rows)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.45))
    fig.subplots_adjust(left=.095, right=.965, bottom=.14, top=.945, wspace=.32, hspace=.43)
    plt.setp(axes.ravel(), facecolor='white')
    for ax, training, mean, sd, ylabel, title in (
        (axes[0, 0], rgb_train, rgb_mean, rgb_sd, 'Relative RGB Reconstruction Error', '(a) RGB Reconstruction Convergence'),
        (axes[0, 1], mean_train, mean_held, sd_held, 'Relative Depth Reconstruction Error', '(b) Depth Reconstruction Convergence')):
        ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[0], alpha=.12, linewidth=0, zorder=1)
        ax.plot(steps, training, color='#71869A', ls='--', lw=1.7, zorder=2)
        ax.plot(steps, mean, color=COLORS[0], lw=2., marker='o', ms=2.4, zorder=3)
        ax.set(xlabel='Training Iteration', ylabel=ylabel, title=title, xlim=(0, steps[-1]), ylim=(0, 1.25))
        ax.set_xticks(np.arange(0, steps[-1] + 1, 1000))
        ax.set_yticks(np.arange(0, 1.26, .25))
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(handles=[Line2D([], [], color='#71869A', ls='--', lw=1.7, label='Training Mean'),
                           Line2D([], [], color=COLORS[0], lw=2, label='Held-out Mean'),
                           Patch(facecolor=COLORS[0], alpha=.12, edgecolor='none', label='Held-out ±1 SD')],
                  frameon=False, loc='upper right', fontsize=7.7, handlelength=2.1, labelspacing=.35)
    lower = {(r['case'], r['stage']): r for r in prior['lower_panel_data']}
    x, width = np.arange(5), .35
    labels = ['00\nRobot', '02\nConveyor', '04\nChiller', '07\nStorage\nCabinet', '10\nTool\nCabinet']
    for ax, key, ylabel, title in (
        (axes[1, 0], 'foreground_psnr_db', 'Foreground PSNR (dB)', '(c) Held-out RGB Reconstruction'),
        (axes[1, 1], 'normalized_depth_mae_percent', 'Normalized Depth MAE (%)', '(d) Held-out Depth Reconstruction')):
        for offset, stage, color in [(-.5, 'Initial', COLORS[4]), (.5, 'PBR optimized', COLORS[0])]:
            ax.bar(x + offset * width, [lower[(case, stage)][key] for case in ORDER], width, label=stage, color=color)
        ax.set_xticks(x, labels)
        ax.set(ylabel=ylabel, title=title)
        if key == 'foreground_psnr_db':
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
    fig.canvas.draw()
    fig.savefig(OUT / 'figure_F12_revised.pdf', facecolor='white')
    fig.savefig(OUT / 'figure_F12_revised.png', dpi=300, facecolor='white')
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(axes.ravel()):
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.04)
        fig.savefig(OUT / 'panels' / f'{chr(97 + i)}.png', dpi=300, bbox_inches=box, facecolor='white')
    plt.close(fig)
    unchanged = {}
    for name in ('a', 'c', 'd'):
        unchanged[name] = np.array_equal(np.asarray(Image.open(OUT / 'panels' / (name + '.png'))),
                                         np.asarray(Image.open(BASE / 'panels' / (name + '.png'))))
        assert unchanged[name], f'Unrequested panel {name} changed'
        shutil.copy2(BASE / 'panels' / (name + '.png'), OUT / 'panels' / (name + '.png'))
    before_image = np.asarray(Image.open(BASE / 'F12.png'))
    after_image = np.asarray(Image.open(OUT / 'figure_F12_revised.png'))
    assert before_image.shape == after_image.shape
    assert np.array_equal(before_image[:, :before_image.shape[1] // 2], after_image[:, :after_image.shape[1] // 2])
    assert np.array_equal(before_image[before_image.shape[0] // 2:], after_image[after_image.shape[0] // 2:])

    caption = (BASE / 'caption.txt').read_text(encoding='utf-8')
    old = ('(a,b) At each common evaluation step, both the training and held-out RGB L1 or absolute depth MAE of each case '
           "are divided by that case's held-out Step-0 value, then averaged with equal weight across all five cases. ")
    new = ('(a) RGB curves average all five cases. (b) Depth curves show a post-hoc leave-Case04-out sensitivity analysis '
           'over Case00, Case02, Case07 and Case10 (n=4). Case04 is excluded from both splits at every step, not only '
           'from selected training observations. Its records are valid and preserved; no measurement or aggregation '
           'error was found. The exclusion was chosen after inspecting the Train/Held-out difference, so (b) is a '
           'four-case subset result, not an unbiased five-case estimate. At each common step, both splits are divided '
           "by the same case's held-out Step-0 value and the included cases receive equal weight. ")
    assert old in caption
    caption = caption.replace(old, new).replace(
        'plus/minus one sample standard deviation of the five normalized held-out errors (n=5, ddof=1).',
        'plus/minus one sample standard deviation of normalized held-out errors (n=5 in a; n=4 in b; ddof=1).')
    (OUT / 'caption.txt').write_text(caption, encoding='utf-8')
    meta = dict(prior)
    meta.update(previous_confirmed_source=source(BASE / 'sources.json'),
                previous_csv=source(BASE / 'multicase_convergence.csv'),
                panel_b={'included_cases': INCLUDE, 'excluded_cases': [EXCLUDE],
                         'reason': 'User-requested post-hoc leave-one-case-out analysis after observing the training/held-out curve order.',
                         'invalid_data_evidence': False, 'original_records_retained': True,
                         'exclude_same_case_all_steps_both_splits': True,
                         'n_cases': 4, 'sample_std_ddof': 1,
                         'final_training_mean': float(mean_train[-1]), 'final_heldout_mean': float(mean_held[-1]),
                         'final_heldout_sd': float(sd_held[-1]), 'audit': audit},
                standard_deviation={'ddof': 1, 'n_by_panel': {'a': 5, 'b': 4}, 'split': 'validation only'},
                final_mean_relative_depth=float(mean_held[-1]),
                final_training_mean_relative_depth=float(mean_train[-1]),
                final_sample_sd_depth=float(sd_held[-1]),
                initial_training_mean_relative_depth=float(mean_train[0]),
                unchanged_panel_pixels=unchanged, only_panel_b_changed=True,
                extra_exclusion_text_inside_figure=False, raw_records_deleted=False,
                script=source(Path(__file__)))
    save(OUT / 'sources.json', meta)
    save(OUT / 'data/aggregation_audit.json', {'cases': audit, 'invalid_metrics_detected': False})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('figure_common.py'), OUT / 'code/figure_common.py')
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1800', '-png', str(OUT / 'figure_F12_revised.pdf'),
                    str(OUT / 'pdf_preview')], check=True)
    Image.open(OUT / 'pdf_preview.png').convert('RGB').save(OUT / 'preview.jpg', quality=95)
    (OUT / 'README.txt').write_text(
        'F12: only panel b revised to four-case leave-Case04-out depth statistics.\n'
        'All original five-case measurements remain in data/five_case_convergence_original.csv.\n'
        'The excluded records are not invalid: raw logs and aggregate means agree.\n'
        'Post-hoc subset scope is disclosed in caption.txt and sources.json; no extra text inside figure.\n'
        'Panels a/c/d and source experiments are unchanged.\n', encoding='utf-8')
    assert before == {p.relative_to(BASE).as_posix(): sha(p) for p in BASE.rglob('*') if p.is_file()}
    print(json.dumps({'output': str(OUT), 'panel_b_n': 4, 'excluded_case': EXCLUDE,
                      'final_training_mean': float(mean_train[-1]), 'final_heldout_mean': float(mean_held[-1]),
                      'unchanged_panels': unchanged}), flush=True)


if __name__ == '__main__':
    main()
