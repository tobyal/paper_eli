"""Restore the user-selected 2000-step dual-mask scene in F10's second row."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import figure_f10_multiview_comparison as layout

ROOT = layout.ROOT
CURRENT = ROOT / 'experiment figure/F10'
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F10_multiview_comparison_dualmask'


def main():
    assert not (OUT / 'sources.json').exists(), 'Preserve a completed correction.'
    for name in ('inputs', 'panels', 'renders', 'code'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    read, write, sha, source = layout.read, layout.write, layout.sha, layout.source
    current = read(CURRENT / 'sources.json')
    accepted_path = CURRENT / 'inputs/previous_F10_sources.json'
    accepted = read(accepted_path)
    cameras = read(CURRENT / 'camera_manifest.json')
    accepted_views = {v['view']: v for v in accepted['selection']}
    selected_scene = Path(accepted['original_sources']['scene']['path'])
    assert selected_scene.name == 'scene_candidate_step2000.ply'
    assert sha(selected_scene) == accepted['original_sources']['scene']['sha256']
    m4 = selected_scene.parent
    composition = read(m4 / 'candidate_composition.json')
    assert sha(selected_scene) == composition['scene_sha256']
    assert sha(m4 / 'object_world.ply') == composition['object_sha256'] == current['sources']['object']['sha256']
    assert sha(Path(accepted['source_cameras']['path'])) == accepted['source_cameras']['sha256']
    raw_cameras = {v['view']: v for v in read(accepted['source_cameras']['path'])['views']}
    full_folder = Path(accepted['source_cameras']['path']).parent / 'full_scene'
    before_hashes = {p.relative_to(CURRENT).as_posix(): sha(p) for p in CURRENT.rglob('*') if p.is_file()}
    for name in ('sources.json', 'caption.txt', 'delivery_manifest.json', 'camera_manifest.json'):
        shutil.copy2(CURRENT / name, OUT / 'inputs' / ('previous_F10_' + name))
    shutil.copy2(accepted_path, OUT / 'inputs/accepted_five_view_sources.json')
    panel_records = {'original': {}, 'reconfigured': {}}
    cached_records = {}
    for view in cameras['views']:
        name = view['view']
        old = accepted_views[name]
        assert view['camera'] == old['original_camera_record']['camera'] == raw_cameras[name]['camera']
        assert view['crop_xyxy'] == old['existing_display_crop']
        original_panel = CURRENT / view['panels']['original']
        original_full = CURRENT / 'renders' / (name + '_original_full.png')
        for src, dst in [(original_panel, OUT / view['panels']['original']),
                         (original_full, OUT / 'renders' / original_full.name)]:
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
        cached = full_folder / (name + '.png')
        old_panel = Path(old['source']['path'])
        assert sha(old_panel) == old['source']['sha256']
        full = Image.open(cached).convert('RGB')
        cropped = full.crop(view['crop_xyxy'])
        assert np.array_equal(np.asarray(cropped), np.asarray(Image.open(old_panel).convert('RGB'))), 'Not the previously confirmed image'
        new_full = OUT / 'renders' / (name + '_reconfigured_full.png')
        new_panel = OUT / view['panels']['reconfigured']
        shutil.copy2(cached, new_full)
        cropped.resize((layout.PANEL_SIZE, layout.PANEL_SIZE), layout.LANCZOS).save(new_panel, dpi=(300, 300))
        assert sha(cached) == sha(new_full)
        cached_records[name] = {'full_render': source(cached), 'confirmed_selected_image': source(old_panel),
                                'same_camera_and_crop': True, 'cropped_pixels_equal_to_confirmed_image': True}
        for state in ('original', 'reconfigured'):
            panel_records[state][name] = {
                'full_render': source(OUT / 'renders' / (name + '_' + state + '_full.png')),
                'panel': source(OUT / view['panels'][state]),
            }
    cameras['source_snapshot'] = 'inputs/accepted_five_view_sources.json'
    write(OUT / 'camera_manifest.json', cameras)
    sources = dict(current['sources'])
    sources['reconfigured'] = source(selected_scene)
    sources['composition'] = source(m4 / 'candidate_composition.json')
    summary = read(m4 / 'background_optimization/summary.json')
    write(OUT / 'render_sources.json', {
        'basis': 'experiment figure/F10; only restore its user-selected dual-mask scene in the reconfigured row',
        'sources': sources, 'panels': panel_records,
        'previous_paired_figure': source(CURRENT / 'F10.png'),
        'previous_wrong_scene': current['sources']['reconfigured'],
        'restored_scene_matches_accepted_five_view_F10': True,
        'cached_render_provenance': cached_records,
        'scene_selection': {
            'policy': 'Explicit user correction: use the existing dual-mask optimized 2000-step result, not the guarded Step-0 fallback.',
            'displayed_step': 2000, 'displayed_candidate': True,
            'candidate_guard_passed': False, 'guarded_selected_step': summary['selected_step'],
            'iterations_in_existing_run': summary['iterations'],
        },
        'original_row_byte_identical': True, 'same_layout_camera_crop': True,
        'no_gaussian_modifications': True, 'no_real_photographs': True,
        'generation_calls': 0, 'training_steps': 0, 'new_gaussian_renders': 0,
        'input_sha256_unchanged': True,
    })
    layout.OUT = OUT
    layout.assemble()
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    assert before_hashes == {p.relative_to(CURRENT).as_posix(): sha(p) for p in CURRENT.rglob('*') if p.is_file()}
    print('Restored confirmed dual-mask scene:', sources['reconfigured']['sha256'], flush=True)


if __name__ == '__main__':
    main()
