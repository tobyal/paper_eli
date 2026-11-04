"""One gradient probe of the actual LangSplat CUDA renderer, not field training."""

import json, math, os, sys
from pathlib import Path
from argparse import ArgumentParser
from types import SimpleNamespace
import torch

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat"))
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from scene.colmap_loader import (
    read_intrinsics_binary,
    read_extrinsics_binary,
    qvec2rotmat,
)
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from gaussian_renderer import render

parser = ArgumentParser()
group = OptimizationParams(parser)
opt = group.extract(parser.parse_args([]))
source, iteration = torch.load(root / "outputs/reconstruction/chkpnt30000.pth")
model = GaussianModel(3)
model.restore(source, opt)
path = root / "data/industrial/sparse/0"
cameras = read_intrinsics_binary(str(path / "cameras.bin"))
images = read_extrinsics_binary(str(path / "images.bin"))
im = next(im for im in images.values() if im.name == "frame_000151.jpg")
intr = cameras[im.camera_id]
fx, fy, cx, cy = intr.params
view = torch.tensor(getWorld2View2(qvec2rotmat(im.qvec).T, im.tvec), device="cuda").T
fovx = focal2fov(fx, intr.width)
fovy = focal2fov(fy, intr.height)
projection = getProjectionMatrix(0.01, 100, fovx, fovy).cuda().T
cam = MiniCam(938, 528, fovy, fovx, 0.01, 100, view, view @ projection)
result = render(
    cam,
    model,
    SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False),
    torch.zeros(3, device="cuda"),
    opt,
)
features = result["language_feature_image"]
assert tuple(features.shape) == (3, 528, 938)
loss = (features - 0.1).square().mean()
loss.backward()
grad = model._language_feature.grad
assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
frozen = all(
    not getattr(model, key).requires_grad and getattr(model, key).grad is None
    for key in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    )
)
assert frozen
record = {
    "probe_only": True,
    "feature_image_shape": list(features.shape),
    "language_gradient_l1": float(grad.abs().sum()),
    "gaussians_with_gradient": int((grad.abs().sum(1) > 0).sum()),
    "geometry_and_appearance_frozen": frozen,
    "optimizer_groups": [g["name"] for g in model.optimizer.param_groups],
}
(root / "records/gsle/language_renderer_validation.json").write_text(
    json.dumps(record, indent=2)
)
print(record, flush=True)
