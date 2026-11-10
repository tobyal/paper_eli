"""Conservative DI-SDS experiment, derived from the pinned native GSLE loop.

Each optimizer update accumulates the complete published random-view batch.
Reference RGB/mask losses are applied once per update, matching a batch mean.
"""

import argparse, math, random, time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from object_model import ObjectGaussians
from diffusion import StableZeroGuidance, DepthInpaintingGuidance
from camera_sampling import sample_views
from provenance import capture
from policy import color_only_optimizer, gradient_norms


def tensor_image(path, channels):
    a = np.asarray(Image.open(path).convert(channels)).copy() / 255
    if a.ndim == 2:
        a = a[:, :, None]
    return torch.as_tensor(a.transpose(2, 0, 1), device="cuda", dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["coarse", "refine"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Explicit diagnostic-only iteration override"
    )
    parser.add_argument(
        "--microbatch",
        type=int,
        help="Physical batch, without changing the effective batch",
    )
    args = parser.parse_args()
    assert args.stage == "refine", "This independent experiment never runs Coarse"
    cfg = config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    stage_cfg = cfg[args.stage]
    dest = OUT / (args.stage + "_diagnostic" if args.steps else args.stage)
    dest.mkdir(exist_ok=True)
    capture(args.stage, dest / f"provenance_{time.time_ns()}.json")
    microbatch = args.microbatch or stage_cfg["batch"]
    assert 1 <= microbatch <= stage_cfg["batch"]
    target = editing_region()
    anchor = json.loads((OUT / "avp/anchor.json").read_text())
    reference = Camera.from_record(anchor["camera"])
    center = np.asarray(target["center"])
    up = np.asarray(target["up"])
    toward = reference.center - center
    horizontal = normalized(toward - up * np.dot(toward, up))
    right = normalized(np.cross(up, horizontal))
    radius = np.linalg.norm(toward)
    ref_elevation = float(np.rad2deg(np.arcsin(np.dot(normalized(toward), up))))
    # Local axes: x=right, y=up, z=toward reference. Det=+1.
    axes = np.stack((right, up, horizontal), axis=1)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    transform = {
        "center": center.tolist(),
        "axes": axes.tolist(),
        "scale": target.get(
            "object_scale",
            float(
                np.max(np.asarray(target["bbox_high"]) - np.asarray(target["bbox_low"]))
                / (2 * cfg["initialization"]["sphere_radius"])
            ),
        ),
    }
    obj = ObjectGaussians(transform, cfg["initialization"])
    first = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda")
    elif args.stage == "refine":
        checkpoint = torch.load(OUT / "coarse/final.pth", map_location="cuda")
    if checkpoint:
        assert (
            checkpoint["transform"] == transform
        ), "Coordinate transform changed between stages"
        obj.load_state(checkpoint["state"])
        if args.resume:
            first = checkpoint["step"]
    optimizer = color_only_optimizer(obj, cfg["optimizer"])
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
    reference_rgb = tensor_image(OUT / "foreground/reference_white.png", "RGB")
    reference_mask = tensor_image(OUT / "foreground/mask.png", "L")
    background = None
    if args.stage == "coarse":
        guidance = StableZeroGuidance(reference_rgb, ref_elevation)
    else:
        guidance = DepthInpaintingGuidance()
        keep = np.load(OUT / "localization/keep_indices.npy")
        background = GaussianData.load(
            ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply",
            selection=keep,
        )
    if args.resume and "rng" in checkpoint:
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    total = args.steps or stage_cfg["iterations"]
    start = time.time()
    attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = dest / ("diagnostic.jsonl" if args.steps else "training.jsonl")
    assert total > first, ("No new optimizer updates requested", first, total)
    save_json(
        dest / "effective_config.json",
        {
            "config": cfg,
            "stage": args.stage,
            "iterations": total,
            "diagnostic_override": args.steps,
            "transform": transform,
            "anchor_camera": reference.record(),
            "effective_batch": stage_cfg["batch"],
            "microbatch": microbatch,
            "gradient_reduction": "mean over random views; one reference loss per optimizer step",
            "teacher_parameters_frozen": True,
            "background_parameters_frozen": True,
            "trainable_parameters": ["color_logit"],
            "frozen_parameters": ["xyz", "log_scale", "rotation", "opacity_logit"],
            "mask_loss_note": "Preserved as in native GSLE; zero parameter gradient when geometry and opacity are frozen.",
        },
    )
    with torch.no_grad():
        save_image(dest / "native_initial.png", render(reference, obj.data(), background=torch.ones(3, device="cuda"), aux=False)["rgb"])
    save_json(dest / "native_initial.json", {"completed_steps": first if args.resume else (cfg["coarse"]["iterations"] if args.stage == "refine" else 0), "stage": args.stage})
    for step in range(first, total):
        tick = time.time()
        obj.schedule(optimizer, stage_cfg, step)
        optimizer.zero_grad(set_to_none=True)
        if args.stage == "coarse":
            level = sum(step >= m for m in stage_cfg["resolution_milestones"]) - 1
            res = stage_cfg["random_resolutions"][level]
            refres = stage_cfg["reference_resolutions"][level]
            rgb_weight = 1500 * step / 1000 if step <= 1000 else 1000.0
            mask_weight = rgb_weight
        else:
            res = refres = 512
            rgb_weight = mask_weight = 1500.0
        cam = reference.resize(refres, refres)
        result = render(cam, obj.data(), background=torch.ones(3, device="cuda"))
        gt_rgb = F.interpolate(
            reference_rgb[None], (refres, refres), mode="bilinear", align_corners=False
        )[0]
        gt_mask = F.interpolate(reference_mask[None], (refres, refres), mode="nearest")[
            0
        ]
        rgb_loss = F.mse_loss(result["rgb"], gt_rgb)
        mask_loss = F.mse_loss(result["alpha"], gt_mask)
        (rgb_weight * rgb_loss + mask_weight * mask_loss).backward()
        reference_values = {
            "rgb_mse": float(rgb_loss.detach()),
            "mask_mse": float(mask_loss.detach()),
        }
        del result, rgb_loss, mask_loss
        sds_value = 0.0
        diagnostics = []
        batch = stage_cfg["batch"]
        views = sample_views(
            cfg["camera_sampling"],
            batch,
            step,
            res,
            center,
            up,
            horizontal,
            right,
            radius,
            transform["scale"],
        )
        if step == 0:
            save_json(
                dest / "first_sampled_cameras.json",
                [
                    {
                        **{k: v for k, v in view.items() if k != "camera"},
                        "camera": view["camera"].record(),
                    }
                    for view in views
                ],
            )
        for first_view in range(0, batch, microbatch):
            rgbs = []
            depths = []
            masks = []
            prompts = []
            elevations = []
            azimuths = []
            for i in range(first_view, min(first_view + microbatch, batch)):
                sample = views[i]
                azimuth = sample["azimuth"]
                elevation = sample["elevation"]
                cam = sample["camera"]
                object_data = obj.data()
                if args.stage == "coarse":
                    result = render(
                        cam,
                        object_data,
                        background=torch.ones(3, device="cuda"),
                        aux=False,
                    )
                else:
                    result = render_joint(cam, background, object_data)
                    masks.append(
                        torch.as_tensor(
                            bbox_mask(cam, np.asarray(target["bbox_corners"])),
                            device="cuda",
                            dtype=torch.float32,
                        )[None]
                    )
                    depths.append(depth_condition(result["depth"], result["alpha"]))
                    view = (
                        "overhead view"
                        if elevation > 60
                        else (
                            "front view"
                            if abs(azimuth) < 45
                            else ("back view" if abs(azimuth) > 135 else "side view")
                        )
                    )
                    prompts.append(cfg["replacement_text"] + ", " + view)
                rgbs.append(result["rgb"].clamp(0, 1))
                elevations.append(elevation)
                azimuths.append(azimuth)
            if args.stage == "coarse":
                loss = guidance.loss_batch(torch.stack(rgbs), elevations, azimuths)
            else:
                loss, diagnostic = guidance.loss_batch(
                    torch.stack(rgbs), torch.stack(depths), torch.stack(masks), prompts
                )
                diagnostics.extend(diagnostic)
            scaled = loss * (stage_cfg["sds_weight"] / batch)
            scaled.backward()
            sds_value += float(loss.detach()) / batch
            del result, loss, scaled, object_data, rgbs, depths, masks
        grads = gradient_norms(obj)
        if not all(math.isfinite(v) for v in grads.values()):
            raise FloatingPointError(("Invalid object gradient", step, grads))
        optimizer.step()
        densified = pruned = 0
        density = cfg["densification"]
        if args.stage == "coarse":
            if (
                step + 1 >= density["start"]
                and step + 1 <= density["end"]
                and (step + 1) % density["period"] == 0
            ):
                densified = obj.densify(
                    optimizer, density["neighbors"], density["max_points_fail"]
                )
            if (
                step + 1 >= density["start"]
                and (step + 1) % density["prune_period"] == 0
            ):
                pruned = obj.prune(optimizer, density["prune_opacity"])
        record = {
            "step": step + 1,
            "stage": args.stage,
            "attempt": attempt,
            "points": len(obj.xyz),
            "seconds": time.time() - tick,
            "elapsed": time.time() - start,
            **reference_values,
            "reference_weight": rgb_weight,
            "sds_loss": sds_value,
            "gradient_norms": grads,
            "random_resolution": res,
            "reference_resolution": refres,
            "densified": densified,
            "pruned": pruned,
            "diffusion": diagnostics,
            "gpu_peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
        }
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if step % 10 == 0:
            print(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("diffusion", "gradient_norms")
                },
                flush=True,
            )
        if step == 0 or (step + 1) % 100 == 0:
            with torch.no_grad():
                save_image(
                    dest / f"previews/reference_{step+1:06d}.png",
                    render(
                        reference,
                        obj.data(),
                        background=torch.ones(3, device="cuda"),
                        aux=False,
                    )["rgb"],
                )
            save_json(RECORDS / f"{args.stage}_progress.json", record)
        if (step + 1) % 500 == 0 or step + 1 == 3500:
            obj.checkpoint(
                dest / f"checkpoint_{step+1:06d}.pth", optimizer, step + 1, args.stage
            )
    tag = "diagnostic" if args.steps else "final"
    obj.checkpoint(dest / f"{tag}.pth", optimizer, total, args.stage)
    obj.data().save(dest / f"{tag}_object_world.ply")
    obj.data(world=False).save(dest / f"{tag}_object_local.ply")
    save_json(
        dest / f"{tag}_summary.json",
        {
            "stage": args.stage,
            "iterations": total,
            "points": len(obj.xyz),
            "elapsed": time.time() - start,
            "diagnostic_override": args.steps,
            "transform": transform,
        },
    )
    print("Finished", args.stage, total, "iterations", flush=True)


if __name__ == "__main__":
    main()
