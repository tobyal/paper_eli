"""Read-only scene audit and export of genuine acquisition/reconstruction data."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiment figure/Experimental_Setup'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    data = OUT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('colmap_io', ROOT / 'repos/gaussian-splatting/scene/colmap_loader.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    sparse = ROOT / 'data/industrial/sparse/0'
    images = io.read_extrinsics_binary(str(sparse / 'images.bin'))
    cameras = io.read_intrinsics_binary(str(sparse / 'cameras.bin'))
    xyz, colors, errors = io.read_points3D_binary(str(sparse / 'points3D.bin'))
    ordered = sorted(images.values(), key=lambda im: int(Path(im.name).stem.split('_')[-1]))
    split = json.loads((ROOT / 'records/camera_split.json').read_text())
    names = [im.name for im in ordered]
    sampled = sorted(p.name for p in (ROOT / 'data/industrial/input').glob('frame_*.jpg'))
    assert names == sampled and set(names) == set(split['train']) | set(split['test'])
    assert not set(split['train']) & set(split['test'])
    assert split['test'] == names[::8]
    assert split['train'] == [n for i, n in enumerate(names) if i % 8]
    rec = ROOT / 'outputs/reconstruction'
    saved_cameras = {c['img_name']: c for c in json.loads((rec / 'cameras.json').read_text())}
    pose_rows = []
    position_errors = []
    for im in ordered:
        R = io.qvec2rotmat(im.qvec)
        center = -R.T @ im.tvec
        forward = R.T @ np.array([0., 0., 1.])
        up = R.T @ np.array([0., -1., 0.])
        position_errors.append(float(np.linalg.norm(center - saved_cameras[Path(im.name).stem]['position'])))
        subset = 'test' if im.name in split['test'] else 'train'
        idx = split[subset].index(im.name)
        camera = cameras[im.camera_id]
        assert camera.model == 'PINHOLE'
        fx, fy, cx, cy = camera.params
        pose_rows.append({'name': im.name, 'sample_id': int(Path(im.name).stem.split('_')[-1]),
                         'image_id': im.id, 'camera_id': im.camera_id, 'split': subset,
                         'qvec_wxyz': im.qvec.tolist(), 'tvec': im.tvec.tolist(), 'R': R.tolist(),
                         'center': center.tolist(), 'forward': forward.tolist(), 'up': up.tolist(),
                         'width': camera.width, 'height': camera.height,
                         'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                         'saved_render': str(rec / subset / 'ours_30000/renders' / f'{idx:05d}.png'),
                         'saved_gt': str(rec / subset / 'ours_30000/gt' / f'{idx:05d}.png')})
    assert max(position_errors) < 1e-5
    counts = {}
    for subset in ('train', 'test'):
        counts[subset] = len(list((rec / subset / 'ours_30000/renders').glob('*.png')))
        assert counts[subset] == len(split[subset])
        assert all(Path(r['saved_render']).exists() and Path(r['saved_gt']).exists()
                   for r in pose_rows if r['split'] == subset)
    save(data / 'camera_poses.json', {'convention': 'COLMAP X_camera = R X_world + t; C_world = -R^T t',
                                     'order': 'numeric sampled frame index; actual fps=2 extraction order',
                                     'cameras': pose_rows})
    # Only thin the sparse cloud for the figure; the complete real trajectory is retained.
    selected = np.random.default_rng(0).choice(len(xyz), min(25000, len(xyz)), replace=False)
    np.savez_compressed(data / 'sparse_cloud_preview.npz', xyz=xyz[selected], rgb=colors[selected],
                        errors=errors[selected], original_indices=selected)
    video = ROOT / 'data/raw/20261016-142321.mp4'
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(video)]))
    save(data / 'video_metadata_current.json', metadata)
    v = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    input_sources = [video, sparse / 'cameras.bin', sparse / 'images.bin', sparse / 'points3D.bin',
                     ROOT / 'records/camera_split.json', rec / 'cfg_args', rec / 'cameras.json',
                     ROOT / 'scripts/01_preprocess.sh', ROOT / 'scripts/05_reconstruction.sh',
                     ROOT / 'repos/gaussian-splatting/render.py',
                     rec / 'point_cloud/iteration_30000/point_cloud.ply']
    sources = [source(p) for p in input_sources]
    assert sources[0]['sha256'] == '061413776d7e163c53aa37cd556018bd65356df2128f4929a55ed90645c67116'
    header = []
    with open(input_sources[-1], 'rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header': break
    n_gaussians = int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
    report = {'video_duration_seconds': float(v['duration']), 'container_duration_seconds': float(metadata['format']['duration']),
              'video_width': v['width'], 'video_height': v['height'], 'video_frames': int(v['nb_frames']),
              'video_nominal_fps': v['r_frame_rate'], 'sampling_fps': 2,
              'sampled_views': len(sampled), 'registered_views': len(names),
              'training_views': len(split['train']), 'heldout_views': len(split['test']),
              'saved_render_counts': counts, 'sparse_points': len(xyz), 'gaussians': n_gaussians,
              'reconstruction_iterations': 30000,
              'max_camera_position_disagreement': max(position_errors),
              'acquisition_hardware': None, 'metric_scale_calibration': None,
              'cases': ['case00', 'case02', 'case04', 'case07', 'case10'],
              'sources': sources}
    save(data / 'dataset_audit.json', report)
    picks = np.rint(np.linspace(0, len(ordered) - 1, 24)).astype(int)
    review = Image.new('RGB', (1920, 6 * 300), 'white')
    draw = ImageDraw.Draw(review)
    review_sources = []
    for i, idx in enumerate(picks):
        row = pose_rows[int(idx)]
        raw = ROOT / 'data/industrial/input' / row['name']
        undistorted = ROOT / 'data/industrial/images' / row['name']
        for directory, path in [('sampled_rgb', raw), ('registered_rgb', undistorted),
                                ('saved_renders', Path(row['saved_render']))]:
            dst = data / directory / row['name'].replace('.jpg', '.png' if directory == 'saved_renders' else '.jpg')
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dst)
        rgb = Image.open(raw).convert('RGB').resize((480, 270), Image.Resampling.LANCZOS)
        review.paste(rgb, ((i % 4) * 480, (i // 4) * 300))
        draw.text(((i % 4) * 480 + 8, (i // 4) * 300 + 277), row['name'], fill='black')
        review_sources.append({'sample_id': row['sample_id'], 'raw': source(raw),
                               'registered_rgb': source(undistorted), 'render': source(row['saved_render'])})
    review.save(data / 'candidate_frames.jpg', quality=94)
    save(data / 'candidate_sources.json', review_sources)
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2))


if __name__ == '__main__': main()
