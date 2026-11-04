"""Shared paths, calibrated cameras, raw Gaussian data and artifact conventions."""

import json, math, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(os.environ.get("ROOT", "/mnt/sda/ubuntu/lkx/papers/paper_eli"))
OUT = Path(os.environ.get("GSLE_OUTPUT", ROOT / "outputs/gsle"))
RECORDS = Path(os.environ.get("GSLE_RECORDS", ROOT / "records/gsle"))
if OUT != ROOT / "outputs/gsle" and RECORDS == ROOT / "records/gsle":
    raise ValueError(
        "A separate GSLE_OUTPUT requires a separate GSLE_RECORDS directory"
    )
sys.path.insert(0, str(ROOT / "repos/gaussian-splatting"))
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    qvec2rotmat,
)
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import eval_sh, RGB2SH
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def config():
    return json.loads(
        Path(os.environ.get("GSLE_CONFIG", ROOT / "gsle/config.json")).read_text()
    )


def editing_region():
    path = OUT / "localization/edit_region.json"
    return json.loads(
        (path if path.exists() else OUT / "localization/target.json").read_text()
    )


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_image(path, image):
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def normalized(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)


def load_calibration():
    path = ROOT / "data/industrial/sparse/0"
    return read_intrinsics_binary(str(path / "cameras.bin")), read_extrinsics_binary(
        str(path / "images.bin")
    )


class Camera:
    """COLMAP world-to-camera, camera +x right,+y down,+z forward."""

    def __init__(self, R, t, width, height, fx, fy, cx=None, cy=None, name="virtual"):
        self.R = np.asarray(R, dtype=np.float32)
        self.t = np.asarray(t, dtype=np.float32)
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(width / 2 if cx is None else cx)
        self.cy = float(height / 2 if cy is None else cy)
        self.name = name
        self.center = -self.R.T @ self.t

    @classmethod
    def from_record(cls, r):
        return cls(
            **{
                k: r[k]
                for k in ("R", "t", "width", "height", "fx", "fy", "cx", "cy", "name")
            }
        )

    def record(self):
        return {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "name": self.name,
        }

    def resize(self, w, h):
        return Camera(
            self.R,
            self.t,
            w,
            h,
            self.fx * w / self.width,
            self.fy * h / self.height,
            self.cx * w / self.width,
            self.cy * h / self.height,
            self.name,
        )

    def project(self, xyz):
        p = xyz @ self.R.T + self.t
        z = p[:, 2]
        return (
            np.stack(
                (p[:, 0] / z * self.fx + self.cx, p[:, 1] / z * self.fy + self.cy),
                axis=-1,
            ),
            z,
        )

    def matrices(self, device):
        view = torch.eye(4, device=device)
        view[:3, :3] = torch.as_tensor(self.R, device=device)
        view[:3, 3] = torch.as_tensor(self.t, device=device)
        proj = getProjectionMatrix(
            0.01,
            100,
            2 * math.atan(self.width / (2 * self.fx)),
            2 * math.atan(self.height / (2 * self.fy)),
        ).to(device)
        proj[0, 2] = 2 * self.cx / self.width - 1
        proj[1, 2] = 2 * self.cy / self.height - 1
        return view.T.contiguous(), (proj @ view).T.contiguous()


def look_at(position, target, up, resolution=512, fov=49.1, name="orbit"):
    forward = normalized(np.asarray(target) - position)
    right = normalized(np.cross(forward, up))
    down = np.cross(forward, right)
    R = np.stack((right, down, forward))
    f = resolution / (2 * np.tan(np.deg2rad(fov) / 2))
    return Camera(R, -R @ position, resolution, resolution, f, f, name=name)


def source_cameras(width=938):
    intr, ext = load_calibration()
    result = []
    for im in sorted(ext.values(), key=lambda x: x.name):
        c = intr[im.camera_id]
        w = width
        h = 2 * round(c.height * w / c.width / 2)
        fx, fy, cx, cy = c.params
        result.append(
            Camera(
                qvec2rotmat(im.qvec),
                im.tvec,
                w,
                h,
                fx * w / c.width,
                fy * h / c.height,
                cx * w / c.width,
                cy * h / c.height,
                im.name,
            )
        )
    return result


class GaussianData:
    def __init__(self, xyz, log_scale, rotation, opacity_logit, sh):
        self.xyz, self.log_scale, self.rotation, self.opacity_logit, self.sh = (
            xyz,
            log_scale,
            rotation,
            opacity_logit,
            sh,
        )

    @classmethod
    def load(cls, path, device="cuda", selection=None):
        v = PlyData.read(str(path))["vertex"].data
        if selection is not None:
            v = v[selection]

        def t(names):
            return torch.as_tensor(
                np.stack([v[n] for n in names], axis=1).copy(), device=device
            )

        dc = t(["f_dc_" + str(i) for i in range(3)])[:, :, None]
        rest = sorted(
            [n for n in v.dtype.names if n.startswith("f_rest_")],
            key=lambda n: int(n.rsplit("_", 1)[1]),
        )
        sh = (
            torch.cat((dc, t(rest).reshape(-1, 3, len(rest) // 3)), dim=2)
            if rest
            else dc
        )
        return cls(
            t(["x", "y", "z"]),
            t(["scale_" + str(i) for i in range(3)]),
            t(["rot_" + str(i) for i in range(4)]),
            t(["opacity"]),
            sh,
        )

    def subset(self, idx):
        return GaussianData(
            *(
                getattr(self, n)[idx]
                for n in ("xyz", "log_scale", "rotation", "opacity_logit", "sh")
            )
        )

    def colors(self, center):
        degree = round(math.sqrt(self.sh.shape[-1]) - 1)
        direction = torch.nn.functional.normalize(
            self.xyz - torch.as_tensor(center, device=self.xyz.device), dim=-1
        )
        return (eval_sh(degree, self.sh, direction) + 0.5).clamp_min(0)

    def save(self, path):
        n = len(self.xyz)
        a = [
            self.xyz,
            torch.zeros_like(self.xyz),
            self.sh[:, :, 0],
            self.sh[:, :, 1:].reshape(n, -1),
            self.opacity_logit,
            self.log_scale,
            self.rotation,
        ]
        values = torch.cat(a, dim=1).detach().float().cpu().numpy()
        names = (
            ["x", "y", "z", "nx", "ny", "nz"]
            + ["f_dc_" + str(i) for i in range(3)]
            + ["f_rest_" + str(i) for i in range((self.sh.shape[-1] - 1) * 3)]
            + ["opacity"]
            + ["scale_" + str(i) for i in range(3)]
            + ["rot_" + str(i) for i in range(4)]
        )
        array = np.empty(n, dtype=[(k, "f4") for k in names])
        array.view(np.float32).reshape(n, len(names))[:] = values
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(array, "vertex")]).write(str(path))


def raster(camera, data, colors, background=None):
    device = data.xyz.device
    background = torch.zeros(3, device=device) if background is None else background
    view, projection = camera.matrices(device)
    settings = GaussianRasterizationSettings(
        image_height=camera.height,
        image_width=camera.width,
        tanfovx=camera.width / (2 * camera.fx),
        tanfovy=camera.height / (2 * camera.fy),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=projection,
        sh_degree=0,
        campos=torch.as_tensor(camera.center, device=device),
        prefiltered=False,
        debug=False,
    )
    screen = torch.zeros_like(data.xyz, requires_grad=True)
    result = GaussianRasterizer(settings)(
        means3D=data.xyz,
        means2D=screen,
        shs=None,
        colors_precomp=colors,
        opacities=data.opacity_logit.sigmoid(),
        scales=data.log_scale.exp(),
        rotations=torch.nn.functional.normalize(data.rotation, dim=-1),
        cov3D_precomp=None,
    )
    return result[0]


def render(camera, data, object_flags=None, background=None, aux=True):
    rgb = raster(camera, data, data.colors(camera.center), background)
    result = {"rgb": rgb}
    if aux:
        d = torch.linalg.norm(
            data.xyz - torch.as_tensor(camera.center, device=data.xyz.device), dim=-1
        )
        flags = torch.zeros_like(d) if object_flags is None else object_flags
        packed = raster(
            camera, data, torch.stack((d, torch.ones_like(d), flags), dim=1)
        )
        result.update(depth=packed[0:1], alpha=packed[1:2], object_alpha=packed[2:3])
    return result


def render_joint(camera, background, object_data, aux=True):
    """Exact joint visibility, with fixed-background shading outside autograd."""
    fields = ("xyz", "log_scale", "rotation", "opacity_logit")
    joint = GaussianData(
        *(torch.cat((getattr(background, k), getattr(object_data, k))) for k in fields),
        sh=None,
    )
    with torch.no_grad():
        background_colors = background.colors(camera.center)
    colors = torch.cat((background_colors, object_data.colors(camera.center)))
    result = {"rgb": raster(camera, joint, colors)}
    if aux:
        # DI-SDS treats diffusion conditions as fixed in the score estimate.
        with torch.no_grad():
            d = torch.linalg.norm(
                joint.xyz - torch.as_tensor(camera.center, device=joint.xyz.device),
                dim=-1,
            )
            packed = raster(
                camera,
                joint,
                torch.stack((d, torch.ones_like(d), torch.zeros_like(d)), dim=1),
            )
        result.update(depth=packed[0:1], alpha=packed[1:2])
    return result


def bbox_corners(low, high):
    return np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )


def bbox_mask(camera, corners):
    import cv2

    xy, z = camera.project(corners)
    mask = np.zeros((camera.height, camera.width), np.uint8)
    if (z > 0).all():
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(xy).astype(np.int32)), 1)
    return mask


def depth_condition(depth, alpha):
    """Paper Eq2 weighted Euclidean depth, normalized then inverted (1-d)."""
    valid = alpha > 0.01
    values = depth[valid]
    if not len(values):
        raise ValueError("No visible scene depth")
    low, high = values.min(), values.max()
    return (
        torch.where(valid, 1 - (depth - low) / (high - low).clamp_min(1e-6), 0)
        .clamp(0, 1)
        .repeat(3, 1, 1)
    )


def concatenate(a, b):
    assert a.sh.shape[-1] == b.sh.shape[-1]
    return GaussianData(
        *(
            torch.cat((getattr(a, k), getattr(b, k)), 0)
            for k in ("xyz", "log_scale", "rotation", "opacity_logit", "sh")
        )
    )
