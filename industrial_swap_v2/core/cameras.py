"""COLMAP world frame; local x right, y up, z toward anchor."""
import numpy as np


def unit(x):
    return np.asarray(x, dtype=float) / max(np.linalg.norm(x), 1e-12)


def region_frame(region, camera):
    center, up = np.asarray(region["center"]), unit(region["up"])
    cam_center = -np.asarray(camera["R"]).T @ np.asarray(camera["t"])
    toward = cam_center - center
    front = unit(toward - up * np.dot(toward, up))
    right = unit(np.cross(up, front))
    axes = np.stack((right, up, front), axis=1)
    assert np.isclose(np.linalg.det(axes), 1)
    corners = (np.asarray(region["bbox_corners"]) - center) @ axes
    return {"axes": axes.tolist(), "center": center.tolist(), "scale": 1.0,
            "local_bbox_low": corners.min(0).tolist(), "local_bbox_high": corners.max(0).tolist(),
            "convention": "world = scale * local @ axes.T + center; local x right, y up, z toward anchor"}


def resized_camera(record, width, height):
    result = dict(record)
    for key in ("fx", "cx"):
        result[key] *= width / record["width"]
    for key in ("fy", "cy"):
        result[key] *= height / record["height"]
    result.update(width=width, height=height)
    return result
