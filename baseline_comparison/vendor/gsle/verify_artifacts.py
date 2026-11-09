"""Verify actual full-run outputs, readable images, Gaussian data and videos."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData


root = Path(os.environ["ROOT"])
out = Path(os.environ.get("GSLE_OUTPUT", root / "outputs/gsle"))
records = Path(os.environ.get("GSLE_RECORDS", root / "records/gsle"))
started = time.time()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


record = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
record["training"] = {}
for stage, expected in (("coarse", 6500), ("refine", 2000)):
    rows = [
        json.loads(line)
        for line in (out / stage / "training.jsonl").read_text().splitlines()
    ]
    latest = {row["step"]: row for row in rows}
    assert set(latest) == set(range(1, expected + 1)), stage
    assert all(
        np.isfinite(list(row["gradient_norms"].values())).all()
        for row in latest.values()
    )
    record["training"][stage] = {
        "updates": expected,
        "finite_saved_gradient_norms": True,
    }

keep = np.load(out / "localization/keep_indices.npy")
removed = np.load(out / "localization/target_indices.npy")
source_path = (
    root / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
source = PlyData.read(str(source_path))["vertex"].data
assert np.array_equal(np.sort(np.concatenate((keep, removed))), np.arange(len(source)))
record["gaussians"] = {}
for relative in (
    "coarse/final_object_world.ply",
    "refine/final_object_local.ply",
    "final/object_world.ply",
    "final/scene.ply",
):
    path = out / relative
    vertices = PlyData.read(str(path))["vertex"].data
    assert len(vertices) > 0
    assert all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names)
    quaternion_norm2 = sum(vertices[f"rot_{i}"] ** 2 for i in range(4))
    assert np.all(quaternion_norm2 > 0), relative
    row = {
        "path": str(path),
        "points": len(vertices),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "finite_all_fields": True,
    }
    if relative == "final/scene.ply":
        assert np.array_equal(source[keep], vertices[: len(keep)])
        row["retained_background_bitwise_identical"] = True
        row["removed_original_points"] = len(removed)
        row["retained_original_points"] = len(keep)
    record["gaussians"][relative] = row

record["render_images"] = {}
for folder, expected, dimensions in (
    ("before", 350, (960, 540)),
    ("after", 350, (960, 540)),
    ("side_by_side", 350, (1920, 540)),
    ("orbit_object", 72, (512, 512)),
    ("orbit_scene", 72, (512, 512)),
):
    paths = sorted((out / "final/renders" / folder).glob("*.png"))
    assert len(paths) == expected, (folder, len(paths))
    for path in paths:
        with Image.open(path) as image:
            assert image.size == dimensions, (path, image.size)
            image.verify()
    record["render_images"][folder] = {
        "count": len(paths),
        "dimensions": dimensions,
        "all_pngs_verified": True,
    }

record["videos"] = {}
for name, frames, fps in (
    ("comparison_original_speed_2fps.mp4", 350, 2),
    ("comparison_6x_preview_12fps.mp4", 350, 12),
    ("edited_original_speed_2fps.mp4", 350, 2),
    ("object_360.mp4", 72, 12),
    ("scene_360.mp4", 72, 12),
):
    path = out / "final/renders" / name
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    assert int(stream["nb_read_frames"]) == frames, name
    assert abs(numerator / denominator - fps) < 1e-6, name
    assert abs(float(stream["duration"]) - frames / fps) < 0.02, name
    record["videos"][name] = {
        **stream,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }

record["source_video_seconds"] = 174.89655
record["original_speed_render_seconds"] = 175.0
record["duration_explanation"] = (
    "350 calibrated samples at 2fps; 12fps is explicitly a 6x preview. The small duration difference is frame sampling/encoding rounding."
)
record["quality_scope"] = (
    "Integrity and full execution verification, not proof of visual or mechanical correctness. See inspected matching views and GSLE report."
)
record["seconds"] = time.time() - started
(records / "final_artifact_verification.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
