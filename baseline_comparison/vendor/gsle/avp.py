"""GSLE anchor proposal: 100 azimuths, masked HSV contrast, eight rotations.

Inputs: localization/target.json + target_indices.npy. Save all candidates,
paper-prose and literal-supplement scores, and the selected calibrated view.
"""

import math
import cv2, numpy as np, torch
from common import *


def brightness_scores(rgb, mask):
    value = rgb.max(axis=-1).copy()
    valid = value[mask]
    if len(valid) < 20:
        return None
    value[mask] = (valid - valid.min()) / max(float(valid.max() - valid.min()), 1e-6)
    value[~mask] = 0
    h, w = mask.shape
    ratios = []
    for angle in np.arange(1, 9) * 45:
        transform = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1)
        v = cv2.warpAffine(value, transform, (w, h), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(
            mask.astype(np.uint8), transform, (w, h), flags=cv2.INTER_NEAREST
        ).astype(bool)
        left = v[:, : w // 2][m[:, : w // 2]]
        right = v[:, w // 2 :][m[:, w // 2 :]]
        if not len(left) or not len(right):
            ratios.append(0.5)
            continue
        bl = float(left.mean())
        br = float(right.mean())
        ratios.append(bl / max(bl + br, 1e-8))
    ratios = np.asarray(ratios)
    return {
        "rotated_left_ratios": ratios.tolist(),
        "literal_abs_min_S_minus_1": float(abs(np.min(ratios - 1))),
        "text_contrast": float(np.max(abs(2 * ratios - 1))),
    }


def main():
    cfg = config()
    target = editing_region()
    dest = OUT / "avp"
    dest.mkdir(exist_ok=True)
    ids = np.load(OUT / "localization/target_indices.npy")
    full = GaussianData.load(
        ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
    )
    obj = full.subset(ids)
    flags = torch.zeros(len(full.xyz), device="cuda")
    flags[torch.as_tensor(ids, device="cuda")] = 1
    center = np.array(target["center"])
    up = np.array(target["up"])
    ref = Camera.from_record(target["reference_camera"])
    toward = ref.center - center
    horizontal = normalized(toward - up * np.dot(toward, up))
    right = normalized(np.cross(up, horizontal))
    elevation = float(
        np.clip(np.rad2deg(np.arcsin(np.dot(normalized(toward), up))), 10, 45)
    )
    corners = np.asarray(target["bbox_corners"])
    sphere_radius = np.linalg.norm(corners - center, axis=1).max()
    fov = 49.1
    radius = float(sphere_radius / math.sin(math.radians(fov / 2)) * 1.15)
    radius *= float(cfg["avp"].get("radius_multiplier", 1.0))
    elevation = float(np.clip(elevation + cfg["avp"].get("elevation_offset_degrees", 0.0), 10, 65))
    records = []
    with torch.no_grad():
        for i in range(cfg["avp"]["views"]):
            angle = 2 * math.pi * i / cfg["avp"]["views"]
            e = math.radians(elevation)
            position = center + radius * (
                math.cos(e) * (math.cos(angle) * horizontal + math.sin(angle) * right)
                + math.sin(e) * up
            )
            cam = look_at(position, center, up, 512, fov, f"avp_{i:03d}")
            joined = render(cam, full, flags)
            isolated = render(cam, obj)
            mask = (isolated["alpha"][0] > 0.5).cpu().numpy()
            rgb = isolated["rgb"].permute(1, 2, 0).cpu().numpy()
            score = brightness_scores(rgb, mask)
            visible = float(
                joined["object_alpha"].sum() / isolated["alpha"].sum().clamp_min(1e-6)
            )
            full_coverage = float((joined["alpha"] > 0.5).float().mean())
            record = {
                "index": i,
                "azimuth": i * 360 / cfg["avp"]["views"],
                "elevation": elevation,
                "radius": radius,
                "visibility": visible,
                "scene_coverage": full_coverage,
                "camera": cam.record(),
                "score": score,
                "eligible": score is not None
                and visible >= cfg["avp"]["min_visibility"]
                and full_coverage >= cfg["avp"]["min_scene_coverage"],
            }
            records.append(record)
            save_image(dest / f"candidates/{i:03d}.png", joined["rgb"])
            save_image(dest / f"candidate_masks/{i:03d}.png", mask)
            if i % 10 == 0:
                print(
                    "AVP",
                    i,
                    "visibility",
                    visible,
                    "coverage",
                    full_coverage,
                    flush=True,
                )
    eligible = [r for r in records if r["eligible"]]
    save_json(dest / "candidates.json", records)
    if not eligible:
        raise RuntimeError(
            "No visible AVP candidate; inspect saved geometry/cameras before adapting the automatic orbit"
        )
    literal = min(eligible, key=lambda r: r["score"]["literal_abs_min_S_minus_1"])
    contrast = max(eligible, key=lambda r: r["score"]["text_contrast"])
    chosen = contrast if cfg["avp"]["scoring"] == "text_contrast" else literal
    camera = Camera.from_record(chosen["camera"])
    with torch.no_grad():
        result = render(camera, full, flags)
        conditioning = depth_condition(result["depth"], result["alpha"])
    mask = bbox_mask(camera, corners)
    save_image(dest / "anchor.png", result["rgb"])
    save_image(dest / "bbox_mask.png", mask)
    save_image(dest / "depth_control.png", conditioning)
    np.savez_compressed(
        dest / "depth.npz",
        depth=result["depth"].cpu().numpy(),
        alpha=result["alpha"].cpu().numpy(),
        object_alpha=result["object_alpha"].cpu().numpy(),
    )
    metadata = {
        "selected": chosen,
        "literal_selected_index": literal["index"],
        "text_selected_index": contrast["index"],
        "scoring": cfg["avp"]["scoring"],
        "eligible_views": len(eligible),
        "near_plane": 0.01,
        "far_plane": 100.0,
        "depth_definition": "volumetric Euclidean distance, normalize then 1-d",
        "camera": camera.record(),
    }
    save_json(dest / "anchor.json", metadata)
    save_json(RECORDS / "avp.json", metadata)
    print(
        "Selected AVP",
        chosen["index"],
        "literal alternative",
        literal["index"],
        flush=True,
    )


if __name__ == "__main__":
    main()
