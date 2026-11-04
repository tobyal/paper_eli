"""306 existing training views; full calibrated images, no manual target ROI."""

import importlib.util, json, os
from pathlib import Path

root = Path(os.environ["ROOT"])
spec = importlib.util.spec_from_file_location(
    "colmap_loader", root / "repos/gaussian-splatting/scene/colmap_loader.py"
)
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
source = root / "data/industrial"
dest = root / "data/gsle_langsplat"
cameras = io.read_intrinsics_binary(str(source / "sparse/0/cameras.bin"))
images = io.read_extrinsics_binary(str(source / "sparse/0/images.bin"))
split = json.loads((root / "records/camera_split.json").read_text())
selected = sorted(
    [im for im in images.values() if im.name in split["train"]], key=lambda im: im.name
)
(dest / "images").mkdir(parents=True, exist_ok=True)
(dest / "sparse/0").mkdir(parents=True, exist_ok=True)
camera_lines = []
for c in cameras.values():
    assert c.model == "PINHOLE"
    camera_lines.append(
        f"{c.id} PINHOLE {c.width} {c.height} "
        + " ".join(f"{v:.17g}" for v in c.params)
        + "\n"
    )
(dest / "sparse/0/cameras.txt").write_text("".join(camera_lines))
lines = []
for im in selected:
    pose = " ".join(f"{v:.17g}" for v in list(im.qvec) + list(im.tvec))
    lines.append(f"{im.id} {pose} {im.camera_id} {im.name}\n\n")
    link = dest / "images" / im.name
    if not link.exists():
        link.symlink_to(source / "images" / im.name)
(dest / "sparse/0/images.txt").write_text("".join(lines))
if not (dest / "sparse/0/points3D.ply").exists():
    (dest / "sparse/0/points3D.ply").symlink_to(source / "sparse/0/points3D.ply")
record = {
    "source": str(source),
    "dataset": str(dest),
    "train": len(selected),
    "held_out": len(split["test"]),
    "image_resize": [938, 528],
    "feature_levels": [1, 2, 3],
    "base_checkpoint": str(root / "outputs/reconstruction/chkpnt30000.pth"),
    "manual_localization_used": False,
}
(root / "records/gsle/dataset.json").write_text(json.dumps(record, indent=2))
print(record)
