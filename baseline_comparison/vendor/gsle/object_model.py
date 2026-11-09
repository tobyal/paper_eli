"""New object Gaussians; GSGen compactness densification, frozen world transform.

Compactness formula ported from GSGen commit e60994e, gaussian_splatting.py
634-697 and utils/ops.py 137-153. cKDTree replaces the exact Euclidean KNN
implementation; no change to the surface-distance or gap-filling formula.
"""

import math, random
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from common import GaussianData, RGB2SH


def quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def rotation_matrix(q):
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * w * x,
            2 * x * z - 2 * w * y,
            2 * y * z + 2 * w * x,
            1 - 2 * x * x - 2 * y * y,
        ),
        -1,
    ).reshape(-1, 3, 3)


def surface_distance(mean, scale, rotation, query):
    xyz = torch.einsum("bij,bj->bi", rotation.transpose(-1, -2), query - mean)
    xyz = torch.nn.functional.normalize(xyz, dim=-1)
    x, y, z = xyz.unbind(-1)
    rxy = torch.sqrt(x * x + y * y + 1e-10)
    d2 = scale[:, 0] ** 2 * (x / rxy) ** 2 + scale[:, 1] ** 2 * (y / rxy) ** 2
    # Preserve GSGen's d2**2 exactly, including its unusual exponent.
    return torch.sqrt(scale[:, 2] ** 2 * z**2 + d2**2 * rxy**2 + 1e-10)


class ObjectGaussians(torch.nn.Module):
    fields = ("xyz", "log_scale", "rotation", "opacity_logit", "color_logit")

    def __init__(self, transform, initialization, device="cuda"):
        super().__init__()
        self.transform = transform
        center = torch.tensor(transform["center"], device=device, dtype=torch.float32)
        axes = torch.tensor(transform["axes"], device=device, dtype=torch.float32)
        q = Rotation.from_matrix(np.asarray(transform["axes"])).as_quat()
        q = q[[3, 0, 1, 2]]
        self.register_buffer("center", center)
        self.register_buffer("axes", axes)
        self.register_buffer(
            "frame_q", torch.tensor(q, device=device, dtype=torch.float32)
        )
        self.world_scale = float(transform["scale"])
        n = initialization["points"]
        radius = initialization["sphere_radius"]
        xyz = (
            torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
            * torch.rand(n, 1, device=device).pow(1 / 3)
            * radius
        )
        opacity = initialization["opacity_peak"] * torch.exp(
            -0.5
            * (xyz.norm(dim=-1, keepdim=True) / initialization["opacity_radial_std"])
            ** 2
        )
        color = torch.empty(n, 3, device=device).uniform_(
            initialization["color_min"], initialization["color_max"]
        )
        rotation = torch.zeros(n, 4, device=device)
        rotation[:, 0] = 1
        values = (
            xyz,
            torch.full((n, 3), math.log(initialization["scale"]), device=device),
            rotation,
            torch.logit(opacity),
            torch.logit(color),
        )
        for key, value in zip(self.fields, values):
            setattr(self, key, torch.nn.Parameter(value))

    def data(self, world=True):
        xyz = (
            self.xyz @ self.axes.T * self.world_scale + self.center
            if world
            else self.xyz
        )
        scale = self.log_scale + math.log(self.world_scale) if world else self.log_scale
        rotation = (
            quaternion_product(self.frame_q.expand(len(xyz), -1), self.rotation)
            if world
            else self.rotation
        )
        dc = RGB2SH(self.color_logit.sigmoid())[:, :, None]
        sh = torch.cat((dc, torch.zeros(len(xyz), 3, 15, device=xyz.device)), 2)
        return GaussianData(xyz, scale, rotation, self.opacity_logit, sh)

    def optimizer(self, stage_cfg, opt_cfg):
        lr = {
            "xyz": stage_cfg["xyz_lr"][0],
            "log_scale": stage_cfg["scale_lr"][0],
            "rotation": opt_cfg["rotation_lr"],
            "color_logit": opt_cfg["color_lr"],
            "opacity_logit": opt_cfg["opacity_lr"],
        }
        return torch.optim.Adam(
            [
                {"params": [getattr(self, k)], "name": k, "lr": lr[k]}
                for k in self.fields
            ],
            lr=0.0,
            eps=opt_cfg["eps"],
        )

    def schedule(self, optimizer, stage_cfg, step):
        for group in optimizer.param_groups:
            key = {"xyz": "xyz_lr", "log_scale": "scale_lr"}.get(group["name"])
            if key:
                first, last, end = stage_cfg[key]
                fraction = min(step / end, 1)
                group["lr"] = math.exp(
                    math.log(first) * (1 - fraction) + math.log(last) * fraction
                )

    @torch.no_grad()
    def mutate(self, optimizer, keep, new=None):
        for group in optimizer.param_groups:
            key = group["name"]
            old = getattr(self, key)
            values = old[keep]
            if new is not None:
                values = torch.cat((values, new[key]), 0)
            replacement = torch.nn.Parameter(values)
            state = optimizer.state.pop(old, {})
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment in state:
                    state[moment] = state[moment][keep]
                    if new is not None:
                        state[moment] = torch.cat(
                            (state[moment], torch.zeros_like(new[key])), 0
                        )
            group["params"] = [replacement]
            optimizer.state[replacement] = state
            setattr(self, key, replacement)

    @torch.no_grad()
    def densify(self, optimizer, k=3, max_points_fail=1000000):
        xyz = self.xyz
        scale = self.log_scale.exp()
        rot = rotation_matrix(self.rotation)
        _, indices = cKDTree(xyz.detach().cpu().numpy()).query(
            xyz.detach().cpu().numpy(), k=k + 1, workers=8
        )
        indices = torch.as_tensor(indices[:, 1:], device=xyz.device)
        additions = {key: [] for key in self.fields}
        for i in range(k):
            idx = indices[:, i]
            neighbor = xyz[idx]
            di = surface_distance(xyz, scale, rot, neighbor)
            dj = surface_distance(neighbor, scale[idx], rot[idx], xyz)
            distance = (neighbor - xyz).norm(dim=-1)
            mask = di + dj < distance
            direction = (neighbor - xyz) / distance[:, None].clamp_min(1e-10)
            additions["xyz"].append(
                (xyz + direction * (distance + di - dj)[:, None] / 2)[mask]
            )
            additions["log_scale"].append(
                ((distance - di - dj)[mask, None] / 6).expand(-1, 3).log()
            )
            for key in ("rotation", "opacity_logit", "color_logit"):
                additions[key].append(getattr(self, key)[mask])
        new = {key: torch.cat(values) for key, values in additions.items()}
        count = len(new["xyz"])
        if len(xyz) + count > max_points_fail:
            raise RuntimeError(
                "Compactness densification exceeds declared memory limit; inspect rather than silently truncate"
            )
        self.mutate(
            optimizer, torch.ones(len(xyz), device=xyz.device, dtype=torch.bool), new
        )
        return count

    @torch.no_grad()
    def prune(self, optimizer, threshold):
        keep = self.opacity_logit.sigmoid().squeeze(-1) >= threshold
        if keep.sum() < 4:
            raise RuntimeError(
                "Object collapsed: fewer than four Gaussians after opacity pruning"
            )
        count = int((~keep).sum())
        self.mutate(optimizer, keep)
        return count

    def checkpoint(self, path, optimizer, step, stage):
        from pathlib import Path

        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "state": self.state_dict(),
                "transform": self.transform,
                "step": step,
                "stage": stage,
                "optimizer": optimizer.state_dict(),
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            },
            temporary,
        )
        temporary.replace(path)

    def load_state(self, state):
        for key in self.fields:
            setattr(self, key, torch.nn.Parameter(state[key].to(self.center.device)))
        self.load_state_dict(state)
