"""Compare the ported densification against functions extracted from pinned GSGen."""

import ast, types
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
from common import ROOT, config, save_json
from object_model import ObjectGaussians, rotation_matrix


def extract(path, name):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


scope = {"torch": torch, "F": F}
exec(
    compile(
        extract(ROOT / "repos/GSGen/utils/ops.py", "distance_to_gaussian_surface"),
        "pinned_GSGen_ops",
        "exec",
    ),
    scope,
)
exec(
    compile(
        extract(
            ROOT / "repos/GSGen/gs/gaussian_splatting.py",
            "densify_by_compatnes_with_idx",
        ),
        "pinned_GSGen_densify",
        "exec",
    ),
    scope,
)
torch.manual_seed(42)
initial = dict(config()["initialization"])
initial["points"] = 128
obj = ObjectGaussians(
    {"center": [0, 0, 0], "axes": np.eye(3).tolist(), "scale": 1.0}, initial
)
with torch.no_grad():
    obj.rotation.copy_(torch.randn_like(obj.rotation))
    obj.log_scale.add_(torch.randn_like(obj.log_scale) * 0.2)
optimizer = obj.optimizer(config()["coarse"], config()["optimizer"])
loss = sum(getattr(obj, key).square().sum() for key in obj.fields)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
xyz = obj.xyz.detach()
_, idx = cKDTree(xyz.cpu().numpy()).query(xyz.cpu().numpy(), k=4)
reference_idx = torch.cdist(xyz, xyz).topk(4, largest=False).indices[:, 1:]
assert np.array_equal(idx[:, 1:], reference_idx.cpu().numpy())
mock = types.SimpleNamespace(
    mean=obj.xyz,
    svec=obj.log_scale.exp(),
    rotmat=rotation_matrix(obj.rotation),
    qvec=obj.rotation,
    color_before_activation=obj.color_logit,
    alpha_before_activation=obj.opacity_logit,
    svec_inv_act=torch.log,
    pbr=False,
)
expected = [
    scope["densify_by_compatnes_with_idx"](mock, torch.tensor(idx[:, i], device="cuda"))
    for i in (1, 2, 3)
]
expected = {k: torch.cat([row[k] for row in expected]) for k in expected[0]}
before = len(xyz)
added = obj.densify(optimizer, 3)
mapping = {
    "xyz": "mean",
    "log_scale": "svec",
    "rotation": "qvec",
    "color_logit": "color",
    "opacity_logit": "alpha",
}
errors = {
    key: float((getattr(obj, key)[before:] - expected[value]).abs().max())
    for key, value in mapping.items()
}
assert max(errors.values()) < 1e-7, errors
for group in optimizer.param_groups:
    state = optimizer.state[group["params"][0]]
    assert state["exp_avg"].shape == group["params"][0].shape
    assert state["exp_avg"][before:].count_nonzero() == 0
obj.prune(optimizer, 0.01)
save_json(
    ROOT / "records/gsle/compactness_validation.json",
    {
        "pinned_GSGen_source_extracted_at_runtime": True,
        "exact_knn_indices_agree": True,
        "input_gaussians": before,
        "added": added,
        "maximum_errors": errors,
        "adam_moments_extended_and_pruned": True,
    },
)
print("Compactness port matches pinned GSGen", added, errors, flush=True)
