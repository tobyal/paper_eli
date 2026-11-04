"""Check source render agreement and differentiability before any optimization."""

import json
import torch
from common import (
    ROOT,
    OUT,
    GaussianData,
    render,
    source_cameras,
    config,
    save_json,
    save_image,
)
from object_model import ObjectGaussians
from gaussian_renderer import render as upstream_render
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from types import SimpleNamespace

path = ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
data = GaussianData.load(path)
camera = next(c for c in source_cameras(512) if c.name == config()["reference_image"])
official = GaussianModel(3)
official.load_ply(str(path))
view, proj = camera.matrices("cuda")
import math

cam = MiniCam(
    camera.width,
    camera.height,
    2 * math.atan(camera.height / (2 * camera.fy)),
    2 * math.atan(camera.width / (2 * camera.fx)),
    0.01,
    100,
    view,
    proj,
)
with torch.no_grad():
    expected = upstream_render(
        cam,
        official,
        SimpleNamespace(
            convert_SHs_python=False, compute_cov3D_python=False, debug=False
        ),
        torch.zeros(3, device="cuda"),
    )["render"]
    actual = render(camera, data)["rgb"]
    diff = (actual - expected).abs()
    mae = diff.mean().item()
    maximum = diff.max().item()
assert maximum < 1e-4, (mae, maximum)
save_image(OUT / "validation/source_reference.png", actual)
# A small differentiable object at a visible scene location, independent of target localization.
center = camera.center + camera.R[2] * 2
obj = ObjectGaussians(
    {
        "center": center.tolist(),
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "scale": 0.25,
    },
    config()["initialization"],
)
sample = render(camera, obj.data())
loss = (sample["rgb"] ** 2).mean() + sample["alpha"].square().mean()
loss.backward()
grads = {key: float(getattr(obj, key).grad.abs().sum()) for key in obj.fields}
assert all(v > 0 and math.isfinite(v) for v in grads.values()), grads
assert not any(
    getattr(data, key).requires_grad
    for key in ("xyz", "log_scale", "rotation", "opacity_logit", "sh")
)
save_json(
    ROOT / "records/gsle/renderer_validation.json",
    {
        "source_gaussians": len(data.xyz),
        "max_absolute_error": maximum,
        "mean_absolute_error": mae,
        "object_gradient_L1": grads,
        "background_frozen": True,
        "camera": camera.record(),
    },
)
print(
    "Renderer agreement and all five object parameter gradient checks passed",
    mae,
    maximum,
    grads,
    flush=True,
)
