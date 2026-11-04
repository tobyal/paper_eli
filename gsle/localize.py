"""Lift the text-selected LangSplat instance into a Gaussian editing region.

The 3D language field is mandatory. The automatically ranked reference proposal
only disambiguates repeated devices. A robust box is derived from semantically
positive, visible 3D Gaussians, never from the old manual housing ROI.
"""

import cv2
from common import *
from world_up import estimate_up

cfg = config()
dest = OUT / "localization"
query = json.loads((dest / "query.json").read_text())
proposal = query["reference_proposal"]
camera = next(c for c in source_cameras(938) if c.name == cfg["reference_image"])
source = GaussianData.load(
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
xyz = source.xyz.cpu().numpy()
scales = source.log_scale.exp().cpu().numpy()
opacity = source.opacity_logit.sigmoid().cpu().numpy()[:, 0]
scores = []
floor_scores = []
for level in (1, 2, 3):
    data = np.load(dest / f"language_query_level{level}.npz")
    scores.append(data["scores"].max(1))
    floor_scores.append(data["floor_scores"])
# Use the scale automatically selected for the full-object reference proposal.
semantic = scores[proposal["level"] - 1]
segments = np.load(
    ROOT
    / "data/gsle_langsplat/language_features"
    / (Path(cfg["reference_image"]).stem + "_s.npy")
)
reference_mask = segments[proposal["level"]] == proposal["feature_index"]
xy, z = camera.project(xyz)
pixels = np.round(xy).astype(np.int64)
inside = (
    (z > 0.01)
    & (pixels[:, 0] >= 0)
    & (pixels[:, 0] < camera.width)
    & (pixels[:, 1] >= 0)
    & (pixels[:, 1] < camera.height)
)
safe_x = np.clip(pixels[:, 0], 0, camera.width - 1)
safe_y = np.clip(pixels[:, 1], 0, camera.height - 1)
with torch.no_grad():
    rendered = render(camera, source)
distance = np.linalg.norm(xyz - camera.center, axis=-1)
depth = (rendered["depth"] / rendered["alpha"].clamp_min(1e-6))[0].cpu().numpy()
local_distance = depth[safe_y, safe_x]
tolerance = np.maximum(scales.max(1) * 3, 0.02 * np.median(local_distance[inside]))
visible = inside & (abs(distance - local_distance) < tolerance)
# 0.5 is LangSplat's generic-background boundary, not a hand-tuned box cutoff.
seed = visible & reference_mask[safe_y, safe_x] & (semantic > 0.5) & (opacity > 0.05)
assert seed.sum() > 100, (
    "Insufficient 3D language support for selected device",
    int(seed.sum()),
)
low, high = np.quantile(xyz[seed], [0.005, 0.995], axis=0)
margin = (high - low) * 0.05
low -= margin
high += margin
box = ((xyz >= low) & (xyz <= high)).all(1)
up, up_record = estimate_up(
    xyz, scales, opacity, np.max(floor_scores, axis=0), source_cameras()
)
save_json(dest / "world_up.json", up_record)
preserved_floor_count = 0
if "offset" in up_record:
    floor = (abs(xyz @ up + up_record["offset"]) < up_record["tolerance"]) & (
        np.max(floor_scores, axis=0) > semantic
    )
    preserved_floor_count = int((box & floor).sum())
    box &= ~floor
# The editing region is the LangSplat-derived box; retain exact source indices.
indices = np.flatnonzero(box)
keep = np.flatnonzero(~box)
assert 100 < len(indices) < len(xyz) * 0.2, (
    "Unreasonable localization extent",
    len(indices),
)
np.save(dest / "seed_indices.npy", np.flatnonzero(seed))
np.save(dest / "target_indices.npy", indices)
np.save(dest / "keep_indices.npy", keep)
center = (low + high) / 2
corners = bbox_corners(low, high)
target = {
    "text": cfg["target_text"],
    "reference_camera": camera.record(),
    "reference_proposal": proposal,
    "selected_level": proposal["level"],
    "seed_count": int(seed.sum()),
    "target_gaussians": len(indices),
    "retained_gaussians": len(keep),
    "bbox_low": low.tolist(),
    "bbox_high": high.tolist(),
    "bbox_corners": corners.tolist(),
    "center": center.tolist(),
    "up": up.tolist(),
    "up_estimation": up_record,
    "source_ply": str(
        ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
    ),
    "source_indices_preserved": True,
    "manual_roi_used": False,
    "extent_rule": "0.5%/99.5% quantiles of visible text-positive 3D seeds, plus 5% per-axis margin",
    "removal_rule": "Gaussian centers inside the semantic-derived 3D bounding box; preserve coplanar floor-semantic Gaussians when a supported floor plane is available",
    "preserved_floor_gaussians": preserved_floor_count,
    "floor_preservation_applied": "offset" in up_record,
    "semantic_boundary": 0.5,
}
save_json(dest / "target.json", target)
save_json(ROOT / "records/gsle/localization.json", target)
selected = source.subset(indices)
retained = source.subset(keep)
selected.save(dest / "target.ply")
retained.save(dest / "scene_without_target.ply")
with torch.no_grad():
    for cam in [
        c
        for c in source_cameras(938)
        if c.name in (cfg["reference_image"], "frame_000101.jpg", "frame_000201.jpg")
    ]:
        colors = source.colors(cam.center)
        colors[indices] = 0.4 * colors[indices] + 0.6 * torch.tensor(
            [1.0, 0.0, 0.0], device="cuda"
        )
        save_image(
            dest / f"previews/{Path(cam.name).stem}_localized.png",
            raster(cam, source, colors),
        )
        save_image(
            dest / f"previews/{Path(cam.name).stem}_removed.png",
            render(cam, retained, aux=False)["rgb"],
        )
        save_image(
            dest / f"previews/{Path(cam.name).stem}_target.png",
            render(cam, selected, aux=False)["rgb"],
        )
    target_alpha = render(camera, selected)["alpha"][0].cpu().numpy() > 0.3
    iou = float(
        (target_alpha & reference_mask).sum()
        / max((target_alpha | reference_mask).sum(), 1)
    )
save_json(
    dest / "reference_check.json",
    {
        "rendered_target_vs_automatic_reference_mask_iou": iou,
        "manual_ground_truth": False,
    },
)
print(target, "reference IoU", iou, flush=True)
