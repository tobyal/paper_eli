"""Matched original/coarse/refined scene views and actual-run summary.

These are rendered from three PLYs with identical calibrated cameras. Metrics
describe background preservation and training, not unavailable replacement GT.
"""

import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from common import *

dest = OUT / "final/review"
dest.mkdir(parents=True, exist_ok=True)
paths = [
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply",
    OUT / "composition/scene.ply",
    OUT / "final/scene.ply",
]
scenes = [GaussianData.load(path) for path in paths]
anchor = json.loads((OUT / "avp/anchor.json").read_text())
target = editing_region()
reference = Camera.from_record(anchor["camera"])
center = np.asarray(target["center"])
up = np.asarray(target["up"])
offset = reference.center - center
radius = np.linalg.norm(offset)
horizontal = normalized(offset - up * np.dot(offset, up))
right = normalized(np.cross(up, horizontal))
elevation = math.radians(anchor["selected"]["elevation"])
cameras = []
for i in range(8):
    angle = 2 * math.pi * i / 8
    position = center + radius * (
        math.cos(elevation) * (math.cos(angle) * horizontal + math.sin(angle) * right)
        + math.sin(elevation) * up
    )
    cameras.append(look_at(position, center, up, 512, 49.1, f"review_{i:02d}"))
labels = ["Original scene", "Coarse 3D-SDS", "After DI-SDS"]
for page in range(2):
    sheet = Image.new("RGB", (1536, 4 * 540), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(4):
        index = page * 4 + row
        camera = cameras[index]
        for column, scene in enumerate(scenes):
            with torch.no_grad():
                rgb = render(camera, scene, aux=False)["rgb"]
            path = dest / f"views/{index:02d}_{column}.png"
            save_image(path, rgb)
            sheet.paste(Image.open(path), (column * 512, row * 540 + 28))
            draw.text(
                (column * 512 + 8, row * 540 + 7),
                f"{labels[column]} | azimuth {index*45} deg",
                fill="black",
            )
    sheet.save(dest / f"matched_views_{page+1}.jpg", quality=94)
save_json(dest / "cameras.json", [camera.record() for camera in cameras])

# One identical automatic crop makes the localized change easy to inspect.
# All full-frame renders remain available above; no image content is edited.
xy, z = cameras[0].project(np.asarray(target["bbox_corners"]))
assert np.all(z > 0)
low, high = xy.min(0), xy.max(0)
margin = 0.08 * (high - low)
low = np.maximum(np.floor(low - margin), 0).astype(int)
high = np.minimum(np.ceil(high + margin), 512).astype(int)
crop_box = tuple(low.tolist() + high.tolist())
crop_width, crop_height = (high - low).tolist()
hero = Image.new("RGB", (3 * crop_width, crop_height + 28), "white")
draw = ImageDraw.Draw(hero)
for column, label in enumerate(labels):
    crop = Image.open(dest / f"views/00_{column}.png").crop(crop_box)
    hero.paste(crop, (column * crop_width, 28))
    draw.text((column * crop_width + 8, 7), label, fill="black")
hero.save(dest / "anchor_comparison.png")
save_json(
    dest / "anchor_comparison_crop.json",
    {
        "camera_index": 0,
        "same_crop_all_stages": list(crop_box),
        "method": "projected generation bbox plus8% margin; original pixels, no synthesized content",
    },
)

summary = {
    "source_video_seconds": 174.89655,
    "replacement": config()["replacement_text"],
    "target": target,
    "stages": {},
}
figure, axes = plt.subplots(2, 3, figsize=(14, 7))
for stage_index, stage in enumerate(("coarse", "refine")):
    rows = [
        json.loads(line)
        for line in (OUT / stage / "training.jsonl").read_text().splitlines()
    ]
    # Retries can append an already-computed step; retain the latest actual step.
    latest = {row["step"]: row for row in rows}
    ordered = [latest[key] for key in sorted(latest)]
    expected = config()[stage]["iterations"]
    assert len(ordered) == expected and ordered[-1]["step"] == expected
    seconds = np.array([row["seconds"] for row in ordered])
    summary["stages"][stage] = {
        "optimizer_updates": len(ordered),
        "batch": config()[stage]["batch"],
        "logged_step_seconds_sum": float(seconds.sum()),
        "all_attempt_logged_steps": len(rows),
        "all_attempt_logged_step_seconds": sum(row["seconds"] for row in rows),
        "median_step_seconds": float(np.median(seconds)),
        "p90_step_seconds": float(np.quantile(seconds, 0.9)),
        "peak_allocated_GiB": max(row["gpu_peak_GiB"] for row in ordered),
        "final_points": ordered[-1]["points"],
        "last_reference_rgb_mse": ordered[-1]["rgb_mse"],
        "last_reference_mask_mse": ordered[-1]["mask_mse"],
        "attempts": sorted(set(row["attempt"] for row in rows)),
    }
    steps = [row["step"] for row in ordered]
    for key, label in (("rgb_mse", "RGB MSE"), ("mask_mse", "Mask MSE")):
        axes[stage_index, 0].plot(
            steps, [max(row[key], 1e-10) for row in ordered], label=label, linewidth=0.8
        )
    axes[stage_index, 0].set_yscale("log")
    axes[stage_index, 0].legend()
    axes[stage_index, 0].set_title(stage + " reference-view loss")
    axes[stage_index, 1].plot(steps, [row["points"] for row in ordered])
    axes[stage_index, 1].set_title(stage + " new-object Gaussians")
    axes[stage_index, 2].plot(steps, seconds, alpha=0.2, linewidth=0.5)
    medians = [np.median(seconds[max(0, i - 49) : i + 1]) for i in range(len(seconds))]
    axes[stage_index, 2].plot(
        steps, medians, linewidth=1, label="50-step rolling median"
    )
    axes[stage_index, 2].legend()
    axes[stage_index, 2].set_title(stage + " seconds per optimizer update")
    for axis in axes[stage_index]:
        axis.set_xlabel("Stage iteration")
        axis.grid(alpha=0.2)
figure.tight_layout()
figure.savefig(dest / "training_curves.png", dpi=160)
figure.savefig(dest / "training_curves.pdf")
plt.close(figure)
metrics = json.loads((OUT / "final/renders/render_metrics.json").read_text())
split = json.loads((ROOT / "records/camera_split.json").read_text())
summary["background_preservation"] = {}
for group in ("train", "test"):
    rows = [row for row in metrics["views"] if row["camera"] in split[group]]
    values = [
        row["background_mae_outside_old_new_union"]
        for row in rows
        if row["background_mae_outside_old_new_union"] is not None
    ]
    summary["background_preservation"][group] = {
        "views": len(rows),
        "mean_RGB_MAE_outside_edit": float(np.mean(values)),
        "max_view_RGB_MAE_outside_edit": float(np.max(values)),
    }
summary["composition"] = json.loads((OUT / "final/composition.json").read_text())
summary["metric_limits"] = [
    "No ground-truth replacement object or edited scene exists.",
    "Background error excludes the dilated union of old/new object projections.",
    "Multiview geometric quality and text compliance require inspection of the actual orbit renders.",
    "Step timings exclude setup/checkpoint/render overhead; this is a shared GPU server.",
]
save_json(dest / "summary.json", summary)
save_json(RECORDS / "final_summary.json", summary)
print(json.dumps(summary["stages"]), flush=True)
