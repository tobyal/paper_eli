"""Automatic floor/up estimation from the trained language field and geometry."""

import numpy as np


def estimate_up(xyz, scales, opacity, floor_scores, cameras):
    rng = np.random.default_rng(42)
    camera_up = np.mean([-c.R[1] for c in cameras], axis=0)
    camera_up /= np.linalg.norm(camera_up)
    positions = np.array([c.center for c in cameras])
    _, singular, vh = np.linalg.svd(positions - positions.mean(0), full_matrices=False)
    pca_up = vh[-1]
    if pca_up @ camera_up < 0:
        pca_up = -pca_up
    extent = np.median(np.linalg.norm(positions - positions.mean(0), axis=1))
    candidates = np.flatnonzero(
        (floor_scores > 0.55) & (opacity > 0.1) & (scales.min(1) / scales.max(1) < 0.3)
    )
    rejected_plane = None
    if len(candidates) >= 1000:
        chosen = rng.choice(candidates, min(20000, len(candidates)), replace=False)
        points = xyz[chosen]
        tolerance = float(
            np.clip(
                np.median(scales[chosen].min(1)) * 4, extent * 0.0005, extent * 0.01
            )
        )
        best = None
        best_count = 0
        for _ in range(512):
            a, b, c = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal /= length
            if abs(normal @ camera_up) < 0.5:
                continue
            distance = -normal @ a
            inliers = abs(points @ normal + distance) < tolerance
            count = int(inliers.sum())
            if count > best_count:
                best = (normal, distance, inliers)
                best_count = count
        if best:
            rejected_plane = {
                "normal": best[0].tolist(),
                "offset": float(best[1]),
                "inlier_fraction": best_count / len(points),
                "inlier_count": best_count,
                "tolerance": tolerance,
                "normal_camera_up_alignment": float(abs(best[0] @ camera_up)),
            }
        strong_horizontal = (
            best is not None
            and best_count >= 1000
            and best_count / len(points) > 0.2
            and abs(best[0] @ camera_up) > 0.95
        )
        if best and (best_count / len(points) > 0.3 or strong_horizontal):
            # A tight initial threshold can reject noisy reconstructed floors.
            # Refit the supported horizontal plane using a three-tolerance band.
            support = abs(points @ best[0] + best[1]) < 3 * tolerance
            inliers = points[support]
            center = inliers.mean(0)
            _, _, vectors = np.linalg.svd(inliers - center, full_matrices=False)
            normal = vectors[-1]
            if normal @ camera_up < 0:
                normal = -normal
            distance = float(-normal @ center)
            residuals = abs(inliers @ normal + distance)
            preservation_tolerance = max(tolerance, float(np.quantile(residuals, 0.95)))
            return normal, {
                "method": "LangSplat factory-floor query + robust plane fit",
                "normal": normal.tolist(),
                "offset": distance,
                "tolerance": preservation_tolerance,
                "initial_ransac_tolerance": tolerance,
                "initial_inlier_fraction": best_count / len(points),
                "acceptance": "fraction>0.3, or fraction>0.2 with >=1000 inliers and normal/up alignment>0.95; refit within three initial tolerances",
                "semantic_candidates": len(candidates),
                "inlier_fraction": float(support.mean()),
                "camera_pca_singular_values": singular.tolist(),
            }
    reliable = singular[-1] / max(singular[-2], 1e-8) < 0.3 and pca_up @ camera_up > 0.5
    normal = pca_up if reliable else camera_up
    return normal, {
        "method": "camera-center plane" if reliable else "mean camera up fallback",
        "normal": normal.tolist(),
        "semantic_candidates": len(candidates),
        "camera_pca_singular_values": singular.tolist(),
        "rejected_plane": rejected_plane,
    }
