"""Derive upright replacement space from the localized footprint, without clicks.

The old Gaussian deletion set stays in target.json/target_indices.npy. This
separate generation volume permits the requested arm to have a new silhouette.
"""

from common import *

target = json.loads((OUT / "localization/target.json").read_text())
cfg = config()
assert cfg["replacement_region"]["mode"] == "upright_from_original_footprint"
up = np.asarray(target["up"])
old_center = np.asarray(target["center"])
reference = Camera.from_record(target["reference_camera"])
toward = reference.center - old_center
horizontal = normalized(toward - up * np.dot(toward, up))
right = normalized(np.cross(up, horizontal))
axes = np.stack((right, up, horizontal), axis=1)
corners = np.asarray(target["bbox_corners"])
local = corners @ axes
low = local.min(0)
high = local.max(0)
old_sides = high - low
height = float(old_sides.max())
high[1] = low[1] + height
new_corners = bbox_corners(low, high) @ axes.T
center = ((low + high) / 2) @ axes.T
region = {
    **target,
    "center": center.tolist(),
    "bbox_corners": new_corners.tolist(),
    "bbox_low": new_corners.min(0).tolist(),
    "bbox_high": new_corners.max(0).tolist(),
    "object_scale": float(
        (high - low).max() / (2 * cfg["initialization"]["sphere_radius"])
    ),
    "object_scale_definition": "world units per local unit; initial local sphere diameter maps to the largest generation-box side",
    "original_localization": str(OUT / "localization/target.json"),
    "original_bbox_corners": target["bbox_corners"],
    "region_axes": axes.tolist(),
    "original_oriented_side_lengths": old_sides.tolist(),
    "generation_oriented_side_lengths": (high - low).tolist(),
    "generation_region_rule": "preserve original floor-aligned horizontal extents and bottom; raise height to the largest original oriented extent",
    "manual_coordinates": False,
    "paper_status": "explicit industrial replacement adaptation; GSLE does not prescribe new-object box sizing",
}
save_json(OUT / "localization/edit_region.json", region)
save_json(ROOT / "records/gsle/edit_region.json", region)
print(
    json.dumps(
        {
            "original_sides": old_sides.tolist(),
            "generation_sides": (high - low).tolist(),
            "center": center.tolist(),
        }
    ),
    flush=True,
)
