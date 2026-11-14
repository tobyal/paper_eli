"""F15: read existing timers, average valid cases, and document actual reuse paths.

Read-only access to the experiment files and SQLite database; no pipeline imports.
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import statistics
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'industrial_swap_v2'
OLD = BASE / 'outputs/paper_figures_v4/F15_efficiency'
OUT = BASE / 'outputs/paper_figures_v4/F15_computational_efficiency_final'
CASES = ['case00', 'case02', 'case04', 'case07', 'case10']
OPS = [
    ('m1_localization', 'M1', 'M1 Localization'),
    ('m2_import', 'M2', 'M2 Import / SAM'),
    ('trellis', 'M3', 'TRELLIS Inference'),
    ('pbr_render', 'M3', 'PBR Reference Rendering'),
    ('gaussian_train', 'M3', 'Gaussian Training'),
    ('registration', 'M4', 'M4 Registration'),
    ('background', 'M4', 'M4 Background Optimization'),
]
COLORS = {'M1': '#225E9D', 'M2': '#D57A27', 'M3': '#399276', 'M4': '#9A5599'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    for name in ['records', 'code', 'panels']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    provenance, raw, missing, db_snapshot, case_metadata = {}, [], [], {}, []

    def archive(path, relative):
        path = Path(path)
        dest = OUT / 'records' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        provenance[str(path)] = {'sha256': sha(path), 'snapshot': str(dest.relative_to(OUT))}
        return str(path)

    def record(case, op, seconds, source, field, scope):
        assert isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds > 0, (case, op, seconds)
        module = next(x[1] for x in OPS if x[0] == op)
        raw.append(dict(case=case, operation_id=op, module=module, seconds=seconds, minutes=seconds / 60,
                        source=source, field=field, scope=scope))

    old_sources = read(OLD / 'sources.json')
    archive(OLD / 'sources.json', 'prior_F15_sources.json')
    archive(OLD / 'stage_components.csv', 'prior_F15_stage_components.csv')
    archive(OLD / 'figure_F15.pdf', 'prior_F15.pdf')
    archive(ROOT / 'baseline_comparison/prompts.json', 'case_mapping_prompts.json')
    source_cases = {c['case']: c for c in old_sources['cases']}
    assert set(source_cases) == set(CASES)
    db_path = BASE / 'webapp/data/database/app.db'
    connection = sqlite3.connect('file:' + str(db_path) + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    batch_path = BASE / 'outputs/qwen_reselected_04_07_10_m4_2000_v1/batch.json'
    batch = {c['key']: c for c in read(batch_path)['items']}
    archive(batch_path, 'm2_reselected_batch.json')

    for case in CASES:
        c = source_cases[case]
        m3 = Path(c['pbr']['path']).parent
        m1 = c.get('m1_database_record')
        if m1:
            row = connection.execute('SELECT id,status,started_at,finished_at FROM m1_jobs WHERE id=?', (m1['id'],)).fetchone()
            assert row and row['status'] == 'completed'
            row = dict(row)
            assert row == m1
            db_snapshot[case + '_m1'] = row
            seconds = (datetime.fromisoformat(row['finished_at']) - datetime.fromisoformat(row['started_at'])).total_seconds()
            record(case, 'm1_localization', seconds, str(db_path), 'm1_jobs:' + row['id'] + ':finished_at-started_at',
                   'Completed assisted localization job wall time; not the later case-import operation.')
        else:
            missing.append(dict(case=case, operation_id='m1_localization', reason='Original localization timer unavailable; cache-import duration is excluded.'))

        if case in ['case00', 'case02']:
            # Follow the original asset's M3->M2 dependency; do not substitute unrelated reruns.
            m3_id = Path(c['trellis']['path']).parents[2].name
            m3_row = dict(connection.execute('SELECT id,stage,status,inputs FROM stage_runs WHERE id=?', (m3_id,)).fetchone())
            m2_id = json.loads(m3_row['inputs'])['m2']
            stage = dict(connection.execute('SELECT id,stage,action,status,folder,reused_from,started_at,finished_at FROM stage_runs WHERE id=?', (m2_id,)).fetchone())
            timing = dict(connection.execute('SELECT run_id,seconds,status,metadata FROM generation_records WHERE run_id=?', (m2_id,)).fetchone())
            assert stage['status'] == timing['status'] == 'completed' and stage['reused_from'] is None
            assert not json.loads(timing['metadata'] or '{}').get('stage_cache_hit', False)
            generation_path = Path(stage['folder']) / 'outputs/m2/generation.json'
            generation = read(generation_path)
            archive(generation_path, case + '/m2_generation.json')
            db_snapshot[case + '_m2'] = {'source_m3_dependency': m3_row, 'stage': stage, 'generation_record': timing}
            sam = generation.get('sam_called')
            record(case, 'm2_import', timing['seconds'], str(db_path), 'generation_records:run_id=' + m2_id + ':seconds',
                   'M2 worker import branch; SAM executed.' if sam else 'M2 worker import branch; supplied alpha, SAM skipped.')
        else:
            item = batch[case]
            stage = item['stages']['m2']
            assert stage['exit_code'] == 0
            generation_path = m3.parent / 'm2/generation.json'
            generation = read(generation_path)
            archive(generation_path, case + '/m2_generation.json')
            sam = generation.get('sam_called')
            record(case, 'm2_import', stage['seconds'], str(batch_path), 'items[key=' + case + '].stages.m2.seconds',
                   'Completed import_case_image subprocess wall time, including scheduler polling granularity; SAM executed.')

        for key, operation, field, scope in [
            ('trellis', 'trellis', 'inference_seconds', 'Model loading, inference, simplification and native checkpoint save; excludes subsequent GLB export.'),
            ('pbr', 'gaussian_train', 'seconds', 'PBR Gaussian optimization timer, including its recorded evaluations/export; excludes reference rendering.'),
            ('alignment', 'registration', 'seconds', 'Gaussian-aware registration search/refinement and recorded export/checks; excludes M4 background optimization.'),
            ('background', 'background', 'seconds', 'Existing 2000-step local background optimizer timer, including in-loop evaluations/snapshots; excludes registration and mask preparation.'),
        ]:
            path = Path(c[key]['path'])
            assert sha(path) == c[key]['sha256'], ('Existing source changed', path)
            data = read(path)
            archive(path, case + '/' + key + '_' + path.name)
            record(case, operation, data[field], str(path), field, scope)
            if key == 'background':
                assert data['iterations'] == 2000 and data.get('start_step', 0) == 0
            if key == 'pbr':
                assert data['iterations'] == 4000

        cameras_path = m3 / 'pbr_views/cameras.json'
        cameras = read(cameras_path)
        assert len(cameras['views']) == 36
        archive(cameras_path, case + '/pbr_cameras.json')
        record(case, 'pbr_render', cameras['seconds'], str(cameras_path), 'seconds',
               'Complete 36-view Cycles render loop plus RGB/alpha/depth file writing; excludes Blender startup/scene loading.')
        guard_path = Path(c['guard']['path'])
        guard = read(guard_path)
        archive(guard_path, case + '/m4_export_summary.json')
        case_metadata.append({'case': case, 'sam_called': sam, 'pbr_views': len(cameras['views']),
                              'pbr_resolution': cameras['config']['resolution'], 'pbr_devices': cameras.get('devices'),
                              'm3_steps': 4000, 'm4_steps': 2000,
                              'candidate_guard_passed': guard.get('candidate_guard_passed'),
                              'selected_step': guard.get('selected_step'),
                              'note': 'Timer precedes the later manual yaw correction; no correction time is invented.' if case == 'case07' else ''})
    connection.close()
    write_json(OUT / 'records/database_selected_rows.json', {'database': str(db_path), 'read_only': True, 'rows': db_snapshot})
    assert len({(r['case'], r['operation_id']) for r in raw}) == len(raw)
    raw.sort(key=lambda r: (CASES.index(r['case']), [x[0] for x in OPS].index(r['operation_id'])))

    operations = []
    for op, module, name in OPS:
        records = [r for r in raw if r['operation_id'] == op]
        if not records:
            continue
        seconds = [r['seconds'] for r in records]
        operations.append(dict(operation_id=op, module=module, operation=name, n=len(seconds),
                               valid_cases=';'.join(r['case'] for r in records),
                               missing_cases=';'.join(c for c in CASES if c not in {r['case'] for r in records}),
                               mean_seconds=statistics.mean(seconds), mean_minutes=statistics.mean(seconds) / 60,
                               min_seconds=min(seconds), max_seconds=max(seconds),
                               min_minutes=min(seconds) / 60, max_minutes=max(seconds) / 60,
                               aggregation='Arithmetic mean of valid cases; missing excluded.'))
    total = sum(r['mean_seconds'] for r in operations)
    modules = []
    for module in COLORS:
        selected = [r for r in operations if r['module'] == module]
        seconds = sum(r['mean_seconds'] for r in selected)
        modules.append(dict(module=module, recorded_operations=';'.join(r['operation_id'] for r in selected),
                            number_of_operations=len(selected), n_by_operation=';'.join(r['operation_id'] + ':' + str(r['n']) for r in selected),
                            sum_operation_mean_seconds=seconds, sum_operation_mean_minutes=seconds / 60,
                            share_of_recorded_mean_sum_percent=seconds / total * 100,
                            aggregation='Sum of distinct operation means, not measured module/end-to-end latency.'))
    write_csv('runtime_records.csv', raw)
    write_csv('missing_runtime_records.csv', missing)
    write_csv('average_runtime_per_operation.csv', operations)
    write_csv('average_runtime_per_module.csv', modules)

    code_paths = ['scripts/blender_pbr_views.py', 'core/pbr_pipeline.py', 'core/pbr_gaussian_optimizer.py',
                  'core/trellis_worker.py', 'core/gaussian_registration.py', 'core/local_background_optimizer.py',
                  'core/scene_integration.py', 'webapp/adapter.py', 'webapp/worker.py', 'webapp/service.py',
                  'scripts/prepare_reference_alignment.py', 'scripts/run_manual_reselected_batch.py']
    for name in code_paths:
        archive(BASE / name, 'code_evidence/' + name)
    reuse = [
        {'stage': 'M1 Localization', 'module': 'M1', 'full': 'Run', 'reuse': 'Run / verified reuse',
         'evidence': ['webapp/worker.py:matching completed signature + verified outputs', 'scripts/prepare_reference_alignment.py:M1 symlink']},
        {'stage': 'M2 Reference + Mask', 'module': 'M2', 'full': 'Generate / import', 'reuse': 'Reuse / update',
         'evidence': ['webapp/adapter.py:image_import', 'core/scene_integration.py:current M2 reference still required']},
        {'stage': 'TRELLIS Inference', 'module': 'M3', 'full': 'Run', 'reuse': 'Skip',
         'evidence': ['scripts/prepare_reference_alignment.py:reuse completed M3, zero TRELLIS calls']},
        {'stage': 'PBR References', 'module': 'M3', 'full': 'Render', 'reuse': 'Skip',
         'evidence': ['core/pbr_pipeline.py:GLB/HDRI/render-config checks', 'scripts/prepare_reference_alignment.py:M3 directory reuse']},
        {'stage': 'Gaussian Training', 'module': 'M3', 'full': 'Train', 'reuse': 'Reuse trained Gaussian',
         'evidence': ['scripts/prepare_reference_alignment.py:M3 reuse and zero new device training steps', 'core/scene_integration.py:object_refined SHA256 check']},
        {'stage': 'M4 Registration', 'module': 'M4', 'full': 'Run', 'reuse': 'Re-register',
         'evidence': ['core/scene_integration.py:integrate and current generation_frame']},
        {'stage': 'M4 Background Update', 'module': 'M4', 'full': 'Optimize', 'reuse': 'Re-optimize',
         'evidence': ['scripts/prepare_reference_alignment.py:new M4 run with existing M1/M2/M3']},
    ]
    write_json(OUT / 'records/reuse_evidence.json', reuse)
    sources = {
        'cases': CASES, 'selection_policy': 'One original operation record per case from the accepted F15 asset/run lineage; no mixing retries or averaging unrelated revisions.',
        'operations': operations, 'modules': modules, 'recorded_operation_mean_sum_minutes': total / 60,
        'case_metadata': case_metadata, 'database_snapshot': 'records/database_selected_rows.json',
        'files': provenance, 'missing': missing, 'reuse_evidence': reuse,
        'newly_recovered_records': ['Five pbr_views/cameras.json:seconds entries', 'Case00 and Case02 M2 generation_records:seconds via original M3 dependency'],
        'excluded_unmeasured': ['Qwen/manual image generation latency', 'Blender startup and scene loading', 'GLB export after the TRELLIS timer',
                               'Mesh-to-Gaussian initialization', 'M4 dual-mask preparation', 'Uninstrumented outer-process overhead', 'Case07 final manual yaw correction'],
        'no_double_counting': 'Only the seven distinct operation timers; no enclosing M3/M4 stage totals, cumulative progress timers, or per-view render times added again.',
        'comparison_limits': 'Shared RTX A6000 server; historical operation timers have different boundaries. M2 includes supplied-alpha import for Case02 and SAM for four others. No standardized end-to-end or asset-reuse speed benchmark is claimed.',
        'reuse_limits': 'Existing Asset Reuse assumes a completed trained Gaussian with provenance and compatible reference/cache. GLB-only import in the PBR web path still runs Gaussian training. New placement requires current M2 conditions and M4 re-evaluation.',
        'no_training_or_algorithm_execution': True,
    }
    return operations, modules, sources


def plot(operations, modules, sources):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5, 'axes.labelsize': 12.5,
                         'xtick.labelsize': 11.5, 'ytick.labelsize': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    w, h = 13.4, 7.5
    fig = plt.figure(figsize=(w, h), facecolor='white')

    def axis(x, y, width, height):
        return fig.add_axes([x / w, y / h, width / w, height / h])

    def text(x, y, value, **kwargs):
        return fig.text(x / w, y / h, value, **kwargs)

    text(.24, 7.13, '(a) Average Runtime Breakdown', fontsize=16, weight='bold')
    total = sources['recorded_operation_mean_sum_minutes']
    text(13.1, 7.13, f'Sum of recorded step means: {total:.2f} min', ha='right', fontsize=12)
    a = axis(.60, 5.70, 12.25, 1.01)
    left = 0
    for row in modules:
        value = row['sum_operation_mean_minutes']
        name = row['module']
        a.barh(0, value, left=left, height=.53, color=COLORS[name], edgecolor='white', linewidth=1)
        mid = left + value / 2
        label = f'{name}  {value:.2f} min'
        if name in ('M1', 'M2'):
            tx = mid if name == 'M1' else 2.6
            a.annotate(label.replace('  ', '\n'), xy=(mid, .24), xytext=(tx, .68), ha='center', va='center', fontsize=11,
                       arrowprops={'arrowstyle': '-', 'color': COLORS[name], 'lw': .9})
        else:
            a.text(mid, 0, label, color='white', weight='bold', ha='center', va='center', fontsize=14)
        left += value
    a.set(xlim=(-.45, total + .5), ylim=(-.42, 1.0))
    a.set_yticks([])
    a.spines[['left', 'bottom']].set_visible(False)
    a.set_xticks([0, 5, 10, 15, 20])
    a.tick_params(axis='x', length=3)
    a.set_xlabel('Sum of recorded operation means (minutes) - not end-to-end runtime', labelpad=5)

    text(.24, 4.94, '(b) Average Computational Cost per Operation', fontsize=15, weight='bold')
    b = axis(2.98, 1.00, 3.93, 3.65)
    for i, row in enumerate(operations):
        mean, low, high = row['mean_minutes'], row['min_minutes'], row['max_minutes']
        b.barh(i, mean, height=.56, color=COLORS[row['module']], alpha=.91)
        b.errorbar(mean, i, xerr=[[mean - low], [high - mean]], color='#353A40', capsize=3, lw=1.1, fmt='none')
        b.text(high + .13, i, f'{mean:.2f}', va='center', fontsize=11.5)
    b.set_yticks(range(len(operations)), [r['operation'] for r in operations])
    b.invert_yaxis()
    b.set_xlim(0, 9.10)
    b.set_xlabel('Mean Runtime (minutes)', labelpad=7)
    b.xaxis.set_major_locator(MultipleLocator(2))
    b.grid(axis='x', color='#E4E7EB', lw=.7)
    b.set_axisbelow(True)
    b.tick_params(axis='y', length=0, pad=8, labelsize=11.5)
    b.spines['left'].set_visible(False)
    text(4.92, .22, 'Whiskers: observed min-max', ha='center', fontsize=10.5, color='#4E5359')

    text(7.57, 4.94, '(c) Workflow Reusability', fontsize=15, weight='bold')
    c = axis(7.57, 1.0, 5.53, 3.62)
    c.set_xlim(0, 1)
    c.set_ylim(0, 8)
    c.set_axis_off()
    c.text(.59, 7.62, 'Full Equipment\nGeneration', ha='center', va='center', fontsize=11.8, weight='bold')
    c.text(.87, 7.62, 'Existing Asset\nReuse', ha='center', va='center', fontsize=11.8, weight='bold')
    c.plot([0, 1], [7.13, 7.13], color='#CBD0D6', lw=.8)
    for i, row in enumerate(sources['reuse_evidence']):
        y = 6.6 - i
        c.add_patch(Rectangle((0, y - .42), .012, .84, facecolor=COLORS[row['module']], edgecolor='none'))
        c.text(.033, y, row['stage'], va='center', fontsize=10.9)
        c.text(.59, y, row['full'], ha='center', va='center', fontsize=10.8, color='#434A52')
        label = row['reuse'].replace('verified reuse', 'cache').replace('Reuse trained Gaussian', 'Reuse trained\nGaussian')
        c.text(.87, y, label, ha='center', va='center', fontsize=10.6,
               color=COLORS[row['module']] if row['module'] == 'M3' else '#252B32', weight='bold' if row['module'] == 'M3' else 'normal')
        c.plot([0, 1], [y - .48, y - .48], color='#EEF0F3', lw=.6)
    text(7.57, .77, 'Reuse assumes a verified trained Gaussian and matched inputs.', fontsize=10.3, color='#4E5359')
    text(7.57, .53, 'Update M1/M2 for new targets or views; re-run M4 for new placement.', fontsize=10.1, color='#4E5359')
    text(7.57, .29, 'GLB-only reuse still needs PBR training. No reuse runtime is inferred.', fontsize=10.1, color='#4E5359')

    fig.savefig(OUT / 'figure_F15_revised.png', dpi=300)
    fig.savefig(OUT / 'figure_F15_revised.pdf')
    for name, box in [('a_average_runtime_breakdown', (.10, 5.12, 13.2, 2.27)),
                      ('b_average_operation_cost', (.10, .08, 7.18, 5.08)),
                      ('c_workflow_reusability', (7.42, .08, 5.9, 5.08))]:
        fig.savefig(OUT / 'panels' / (name + '.png'), dpi=300, bbox_inches=Bbox.from_bounds(*box))
    plt.close(fig)
    sources['figure'] = {'script': str(Path(__file__)), 'script_sha256': sha(__file__), 'canvas_inches': [w, h],
                         'png_dpi': 300, 'colors': COLORS, 'workflow_order_preserved': True,
                         'sample_sizes_shown_in_panel_b': False, 'sample_sizes_retained_in_csv_and_caption': True,
                         'range_definition': 'Actual minimum and maximum, not confidence intervals.',
                         'source_files_preserved': all(sha(p) == v['sha256'] for p, v in sources['files'].items())}
    assert sources['figure']['source_files_preserved']


def documents(operations, modules, sources):
    total = sources['recorded_operation_mean_sum_minutes']
    caption = (
        'Fig. 15. Recorded computational cost and workflow reusability for five industrial equipment replacement cases '
        '(Case00, Case02, Case04, Case07, and Case10). (a) A single stacked bar sums the arithmetic means of the '
        'distinct recorded operations within M1-M4. Its total of ' + f'{total:.2f}' + ' min is a sum of operation means, '
        'not measured end-to-end latency. (b) Each bar is the arithmetic mean over valid case records; whiskers show '
        'the observed minimum and maximum. M1 has four records; every other displayed operation has five. M2 measures '
        'image import and available foreground processing: Case02 supplies alpha and skips SAM, while the other four '
        'use SAM. PBR references comprise 36 views at 512-pixel resolution; Gaussian training uses 4,000 steps and '
        'background optimization uses 2,000 steps. Timings come from historical operations on a shared RTX A6000 '
        'server, including the overhead within each recorded timer. Qwen/manual generation, several setup/export '
        'steps, and other uninstrumented work are excluded, rather than set to zero. (c) Code-supported reuse of '
        'completed Gaussian assets and verified intermediate results. New placement still requires compatible '
        'reference conditions, registration, and scene-dependent background updates. Importing only a GLB does not '
        'skip PBR Gaussian training. Reuse is a workflow analysis, not a measured speedup. No algorithms were rerun.\n'
    )
    (OUT / 'caption.txt').write_text(caption, encoding='utf-8')
    lines = ['# F15: recorded computational cost and reuse', '',
             'Five cases: Case00, Case02, Case04, Case07, Case10. Existing logs only; no new experiment.', '',
             '## Aggregation', '',
             'For each operation, mean = sum(valid recorded seconds) / number of valid cases. '
             'A missing record is excluded, never zero. One source record per operation per case is retained. '
             'Module cost is the sum of its distinct operation means. The resulting total is **not measured end-to-end latency**.', '',
             '| Operation | n | Mean (min) | Min (min) | Max (min) |', '|---|---:|---:|---:|---:|']
    for row in operations:
        lines.append(f"| {row['operation']} | {row['n']} | {row['mean_minutes']:.6f} | {row['min_minutes']:.6f} | {row['max_minutes']:.6f} |")
    lines += ['', '| Module | Sum of operation means (min) | Share of recorded sum |', '|---|---:|---:|']
    for row in modules:
        lines.append(f"| {row['module']} | {row['sum_operation_mean_minutes']:.6f} | {row['share_of_recorded_mean_sum_percent']:.2f}% |")
    lines += ['', f'**Sum of recorded operation means: {total:.6f} minutes.**', '',
              '## Recovered original records', '',
              '- All five PBR render datasets contain a completed render-loop `seconds` field in `pbr_views/cameras.json`. These were omitted in the previous F15.',
              '- The original M3 asset dependency identifies two additional completed, non-cache-hit M2 import jobs (Case00 and Case02); their actual `generation_records.seconds` are included.',
              '- Case02 imports supplied alpha and does not run SAM. This is a valid M2 import measurement, not a SAM runtime estimate.',
              '- Case00 has no original M1 localization duration. Its later M1 reuse/import is not counted as localization.', '',
              '## Timer boundaries and double-counting controls', '',
              '- TRELLIS includes model loading, inference, simplification, and native save; its subsequent GLB export is outside the saved timer.',
              '- PBR rendering uses the final 36-view `cameras.json.seconds`; per-view log entries and `progress.json` are not added again.',
              '- Gaussian training uses `pbr_training_summary.json.seconds` (4,000 steps). PBR rendering is an earlier, disjoint step.',
              '- Registration and background optimization are disjoint timers; enclosing batch M3/M4 durations are not added.',
              '- M2 imports use worker timers for Case00/02 and completed subprocess timers for Case04/07/10; the latter include scheduler polling granularity.',
              '- M1 uses recorded job start/finish timestamps. No file-mtime differences, guessed latency, or synthetic values are used.',
              '- Case07 timing precedes the later manual yaw adjustment; no adjustment latency is estimated.',
              '- The five M4 candidates completed 2,000 steps even though their preservation guards rejected the candidates. Runtime is computation performed, not evidence of accepted output quality.', '',
              'Unmeasured and excluded: ' + '; '.join(sources['excluded_unmeasured']) + '. Source 3DGS/LangSplat preparation is also outside this per-replacement analysis.', '',
              '## Actual reuse paths', '',
              '- `webapp/worker.py` reuses completed stages only for a matching signature and verified artifact hashes.',
              '- `scripts/prepare_reference_alignment.py` reuses completed M1/M2/M3 directories, including the trained Gaussian and M3 receipt, and creates a separate M4 run without Qwen/TRELLIS or device training.',
              '- `core/scene_integration.py` checks the refined Gaussian SHA256 and uses the current M2 camera, mask and generation space for placement.',
              '- `core/pbr_pipeline.py` reuses compatible PBR view caches by checking GLB/HDRI/config fingerprints; passing only a GLB still proceeds to Gaussian optimization.',
              '- M1/M2 may be reused only when the scene, target and reference conditions remain compatible. A new target/view needs its own preparation; new placement needs M4 masks and optimization.',
              '- No asset-reuse runtime or speedup factor is asserted.', '',
              '## Reproduction and files', '',
              '`envs/gsle/bin/python industrial_swap_v2/scripts/paper_figures_v4/figure_f15_revised.py`', '',
              'The read-only collector follows the original F15 sources. `runtime_records.csv` retains every measured value and exact field; '
              '`records/` contains source JSON snapshots, selected database rows, and code evidence. `sources.json` records SHA256 hashes. '
              'The earlier F15 is preserved.']
    (OUT / 'runtime_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(OUT / 'sources.json', sources)
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)


def main():
    operations, modules, sources = collect()
    plot(operations, modules, sources)
    documents(operations, modules, sources)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2300', '-png',
                    str(OUT / 'figure_F15_revised.pdf'), str(OUT / 'pdf_preview')], check=True)
    for row in operations:
        print(row['operation'], 'n=' + str(row['n']), 'mean_min=' + str(round(row['mean_minutes'], 6)))
    print('SUM_OF_OPERATION_MEANS_MIN', sources['recorded_operation_mean_sum_minutes'])
    print('F15 COMPLETE', OUT)


if __name__ == '__main__':
    main()
