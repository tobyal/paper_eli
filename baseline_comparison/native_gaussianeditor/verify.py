"""Check native-run provenance and real saved artifacts, without GPU work."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    args = p.parse_args()
    out = BASE / 'gaussianeditor_native' / args.case
    config = json.loads((out / 'config.json').read_text())
    prompts = json.loads((out / 'prompts.json').read_text())
    status = json.loads((out / 'records/status.json').read_text())
    assert status['status'] == 'complete'
    forbidden = ['target_geometry', 'industrial_swap_v2', 'native_views', 'ours_source', 'ours_scene']
    for text in [json.dumps(config), json.dumps(prompts)]:
        assert not any(token in text for token in forbidden)
    code = BASE / 'vendor/GaussianEditor_native'
    expected = subprocess.check_output(['git', 'show', config['upstream_commit'] + ':threestudio/utils/sam.py'],
                                       cwd=ROOT / 'repos/GaussianEditor')
    assert (code / 'threestudio/utils/sam.py').read_bytes() == expected
    assert not (code / 'instance_selection.py').exists()
    assert 'GE_TARGET_CAMERA_DEPTH' not in (code / 'webui.py').read_text()
    masks = np.load(out / 'localization/selected_indices.npy')
    localized = json.loads((out / 'records/localization.json').read_text())
    assert len(masks) == localized['gaussians']
    assert localized['total_cameras'] == 350
    region = json.loads((out / 'records/add_region.json').read_text())
    mask = np.asarray(Image.open(out / 'localization_visible_mask.png')) > 0
    yy, xx = np.where(mask)
    lo = np.array([xx.min(), yy.min()]);hi = np.array([xx.max() + 1, yy.max() + 1])
    margin = (hi - lo) * config['add_margin_fraction']
    expected_box = np.r_[np.floor(lo - margin), np.ceil(hi + margin)].astype(int).tolist()
    assert expected_box == region['bbox']
    actual_add = np.asarray(Image.open(out / 'add_region_mask.png')) > 0
    expected_add = np.zeros_like(actual_add)
    x0, y0, x1, y1 = expected_box
    expected_add[y0:y1, x0:x1] = True
    assert np.array_equal(actual_add, expected_add)
    required = ['original_view', 'localization_mask_overlay', 'localization_render_selected',
                'localization_scene_with_box', 'delete_before', 'delete_target_mask', 'delete_after',
                'delete_comparison', 'add_region_overlay', 'add_region_mask', 'add_region_with_deleted_scene',
                'generated_2d_image', 'generated_2d_mask', 'generated_2d_rgba', 'generated_object_3d_preview',
                'scene_inserted_before_refine', 'scene_final', 'scene_final_crop', 'final_comparison']
    for name in required:
        with Image.open(out / (name + '.png')) as image:
            image.verify()
    assert '1500/1500' in (out / 'logs/delete.log').read_text()
    add_log = (out / 'logs/add.log').read_text()
    assert 'max_steps=3000' in add_log and '1801/1801' in add_log
    first = np.asarray(Image.open(out / 'scene_inserted_before_refine.png')).astype(np.int16)
    final = np.asarray(Image.open(out / 'scene_final.png')).astype(np.int16)
    insertion_delta = int(np.abs(first - final).max())
    assert insertion_delta <= 1, 'Bookkeeping/export unexpectedly changed the inserted scene'
    manifest = json.loads((BASE / 'prompts.json').read_text())
    assert sha(Path(config['source_scene'])) == manifest['source_scene_sha256']
    previous = BASE / 'comparison' / args.case / 'gaussianeditor/artifact.json'
    previous_record = json.loads(previous.read_text())
    assert sha(BASE / 'gaussianeditor' / args.case / 'scene_final.ply') == previous_record['sha256']
    artifacts = {}
    for name in ['selected_target.ply', 'scene_deleted.ply', 'add/object_world.ply', 'final_scene.ply']:
        path = out / name
        vertices = PlyData.read(str(path))['vertex'].data
        assert len(vertices) > 0 and all(np.isfinite(vertices[field]).all() for field in vertices.dtype.names)
        artifacts[name] = dict(gaussians=len(vertices), bytes=path.stat().st_size, sha256=sha(path))
    result = dict(status='passed', native_sam_byte_identical=True, no_ours_spatial_inputs=True,
                  add_bbox_verified_from_ge_mask=True, required_images=len(required),
                  source_scene_preserved=True, previous_gaussianeditor_result_preserved=True,
                  post_insertion_refinement=False, insertion_export_max_pixel_delta=insertion_delta, artifacts=artifacts)
    (out / 'records/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
