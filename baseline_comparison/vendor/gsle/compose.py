"""Compose the generated world-space object with unchanged retained Gaussians."""

import argparse, hashlib, time
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["coarse", "refine"])
args = parser.parse_args()
start = time.time()
dest = OUT / ("composition" if args.stage == "coarse" else "final")
dest.mkdir(exist_ok=True)
keep = np.load(OUT / "localization/keep_indices.npy")
source_path = (
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
background = GaussianData.load(source_path, selection=keep)
obj = GaussianData.load(OUT / args.stage / "final_object_world.ply")
combined = concatenate(background, obj)
combined.save(dest / "scene.ply")
obj.save(dest / "object_world.ply")
original = PlyData.read(str(source_path))["vertex"].data[keep]
composed = PlyData.read(str(dest / "scene.ply"))["vertex"].data[: len(keep)]
assert original.dtype.names == composed.dtype.names
assert np.array_equal(
    original, composed
), "Retained background parameters changed during composition"
record = {
    "stage": args.stage,
    "source": str(source_path),
    "source_gaussians": len(keep)
    + len(np.load(OUT / "localization/target_indices.npy")),
    "background_gaussians": len(keep),
    "new_object_gaussians": len(obj.xyz),
    "retained_background_bitwise_identical": True,
    "placement": "already world-space; exact similarity recorded in object checkpoint/effective_config.json",
    "seconds": time.time() - start,
    "scene": str(dest / "scene.ply"),
    "object": str(dest / "object_world.ply"),
}
save_json(dest / "composition.json", record)
print(record, flush=True)
