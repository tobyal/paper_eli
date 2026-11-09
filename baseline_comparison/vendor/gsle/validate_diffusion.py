"""Actual teacher forward/backward probes; these are not editing results."""

import argparse, time
import torch
from common import *
from object_model import ObjectGaussians
from diffusion import DepthInpaintingGuidance, StableZeroGuidance

parser = argparse.ArgumentParser()
parser.add_argument("teacher", choices=["di", "zero"])
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()
torch.manual_seed(42)
start = time.time()
camera = next(
    c for c in source_cameras(512) if c.name == config()["reference_image"]
).resize(512, 512)
source = GaussianData.load(
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
with torch.no_grad():
    original = render(camera, source)
transform = {
    "center": (camera.center + camera.R[2] * 2).tolist(),
    "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "scale": 0.25,
}
obj = ObjectGaussians(transform, config()["initialization"])
if args.teacher == "di":
    guide = DepthInpaintingGuidance()
    with torch.no_grad():
        expected = render(camera, concatenate(source, obj.data()))
        actual = render_joint(camera, source, obj.data())
        assert (expected["rgb"] - actual["rgb"]).abs().max() < 1e-6
        assert (expected["depth"] - actual["depth"]).abs().max() < 1e-6
    result = render_joint(camera, source, obj.data())
    mask = torch.zeros(1, 512, 512, device="cuda")
    mask[:, 128:384, 128:384] = 1
    loss, rows = guide.loss_batch(
        result["rgb"].clamp(0, 1)[None].repeat(args.batch, 1, 1, 1),
        depth_condition(result["depth"], result["alpha"])[None].repeat(
            args.batch, 1, 1, 1
        ),
        mask[None].repeat(args.batch, 1, 1, 1),
        [config()["replacement_text"]] * args.batch,
    )
    diagnostics = rows[0]
    teachers = [
        guide.pipe.vae,
        guide.pipe.unet,
        guide.pipe.text_encoder,
        guide.pipe.controlnet,
        guide.view_text,
    ]
else:
    guide = StableZeroGuidance(original["rgb"].clamp(0, 1), 20.0)
    result = render(
        camera, obj.data(), background=torch.ones(3, device="cuda"), aux=False
    )
    loss = guide.loss_batch(
        result["rgb"][None].repeat(args.batch, 1, 1, 1),
        [30.0] * args.batch,
        [60.0] * args.batch,
    )
    diagnostics = {"use_stable_zero123": guide.guide.use_stable_zero123}
    teachers = [
        guide.guide.pipe.vae,
        guide.guide.pipe.unet,
        guide.guide.pipe.image_encoder,
        guide.guide.pipe.clip_camera_projection,
    ]
loss.backward()
grads = {key: float(getattr(obj, key).grad.abs().sum()) for key in obj.fields}
assert (
    all(np.isfinite(v) for v in grads.values())
    and grads["xyz"] > 0
    and grads["color_logit"] > 0
), grads
assert all(
    p.grad is None and not p.requires_grad
    for teacher in teachers
    for p in teacher.parameters()
)
record = {
    "teacher": args.teacher,
    "batch": args.batch,
    "probe_only": True,
    "seconds": time.time() - start,
    "loss": float(loss.detach()),
    "gradient_L1": grads,
    "teachers_frozen": True,
    "gpu_peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
    **diagnostics,
}
suffix = "" if args.batch == 1 else f"_batch{args.batch}"
save_json(
    ROOT / f"records/gsle/{args.teacher}_gradient_validation{suffix}.json", record
)
print(record, flush=True)
