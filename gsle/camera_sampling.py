"""Paper-era threestudio random-camera strategy mapped into the target frame.

Source and exact revision: camera_source.json. Keep its 50/50 elevation mixture,
stratified azimuth, uniform FOV/radius, and camera/center/up perturbations.
Distance is relative to the data-derived anchor; length perturbations use the
normalized new-object scale. GSLE's baked colors do not need sampled lights.
"""

import math, random
import numpy as np
import torch
from common import look_at


def sample_views(
    cfg, batch, step, resolution, center, up, horizontal, right, radius, object_scale
):
    progress = min(1.0, step / (cfg["progressive_until"] + 1))
    elevation_bounds = [
        (1 - progress) * cfg["eval_elevation"] + progress * v for v in cfg["elevation"]
    ]
    azimuth_bounds = [progress * v for v in cfg["azimuth"]]
    if random.random() < 0.5:
        elevation = (
            torch.rand(batch) * (elevation_bounds[1] - elevation_bounds[0])
            + elevation_bounds[0]
        )
        elevation_rad = torch.deg2rad(elevation)
    else:
        low, high = map(math.radians, elevation_bounds)
        elevation_rad = torch.asin(
            torch.rand(batch) * (math.sin(high) - math.sin(low)) + math.sin(low)
        )
        elevation = torch.rad2deg(elevation_rad)
    azimuth = (
        (torch.rand(batch) + torch.arange(batch)) / batch
        if cfg["stratified_azimuth"]
        else torch.rand(batch)
    ) * (azimuth_bounds[1] - azimuth_bounds[0]) + azimuth_bounds[0]
    azimuth_rad = torch.deg2rad(azimuth)
    low, high = cfg["radius_multiplier"]
    distance = (torch.rand(batch) * (high - low) + low) * radius
    position = torch.stack(
        (
            distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad),
            distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad),
            distance * torch.sin(elevation_rad),
        ),
        dim=-1,
    ).numpy()
    basis = np.stack((horizontal, right, up), axis=1)
    position = center + position @ basis.T
    position += (
        (torch.rand(batch, 3).numpy() * 2 - 1) * cfg["camera_perturb"] * object_scale
    ) @ basis.T
    targets = (
        center
        + (torch.randn(batch, 3).numpy() * cfg["center_perturb"] * object_scale)
        @ basis.T
    )
    ups = up + (torch.randn(batch, 3).numpy() * cfg["up_perturb"]) @ basis.T
    low, high = cfg["fovy"]
    fovy = torch.rand(batch) * (high - low) + low
    return [
        {
            "camera": look_at(
                position[i],
                targets[i],
                ups[i],
                resolution,
                float(fovy[i]),
                f"train_{step}_{i}",
            ),
            "elevation": float(elevation[i]),
            "azimuth": float(azimuth[i]),
            "radius": float(distance[i]),
            "fovy": float(fovy[i]),
        }
        for i in range(batch)
    ]
