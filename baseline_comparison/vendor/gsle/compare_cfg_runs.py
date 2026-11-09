"""Compare completed interpretations using the exact same saved camera views."""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root = Path(os.environ["ROOT"])
out = root / "outputs/gsle"
primary = out / "final/review"
alternative = out / "ablations/text_only_cfg/final/review"
dest = out / "ablation_comparison"
dest.mkdir(parents=True, exist_ok=True)
assert json.loads((primary / "cameras.json").read_text()) == json.loads(
    (alternative / "cameras.json").read_text()
)
design = json.loads(
    (root / "records/gsle/ablations/text_only_cfg/design.json").read_text()
)
labels = (
    "Coarse 3D-SDS",
    "DI-SDS: drop all conditions",
    "DI-SDS: retain context/depth",
)
for page in range(2):
    sheet = Image.new("RGB", (1536, 2160), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(4):
        index = page * 4 + row
        paths = (
            primary / f"views/{index:02d}_1.png",
            primary / f"views/{index:02d}_2.png",
            alternative / f"views/{index:02d}_2.png",
        )
        for column, path in enumerate(paths):
            sheet.paste(Image.open(path), (column * 512, row * 540 + 28))
            draw.text(
                (column * 512 + 6, row * 540 + 7),
                f"{labels[column]} | {index*45}deg",
                fill="black",
            )
    sheet.save(dest / f"matched_views_{page+1}.jpg", quality=94)
crop = json.loads((primary / "anchor_comparison_crop.json").read_text())[
    "same_crop_all_stages"
]
assert (
    crop
    == json.loads((alternative / "anchor_comparison_crop.json").read_text())[
        "same_crop_all_stages"
    ]
)
width, height = crop[2] - crop[0], crop[3] - crop[1]
hero = Image.new("RGB", (width * 3, height + 28), "white")
draw = ImageDraw.Draw(hero)
for column, path in enumerate(
    (
        primary / "views/00_1.png",
        primary / "views/00_2.png",
        alternative / "views/00_2.png",
    )
):
    hero.paste(Image.open(path).crop(tuple(crop)), (column * width, 28))
    draw.text((column * width + 6, 7), labels[column], fill="black")
hero.save(dest / "anchor_comparison.png")
comparison = {"design": design, "matched_camera_count": 8, "variants": {}}
for mode, directory, review in (
    ("drop_all_eq7", out / "refine", primary),
    ("text_only", out / "ablations/text_only_cfg/refine", alternative),
):
    summary = json.loads((review / "summary.json").read_text())
    assert summary["stages"]["refine"]["optimizer_updates"] == 2000
    rows = [
        json.loads(line)
        for line in (directory / "training.jsonl").read_text().splitlines()
    ]
    rows = list({row["step"]: row for row in rows}.values())
    residuals = [
        entry["noise_residual_rms"] for row in rows for entry in row["diffusion"]
    ]
    render_metrics = json.loads(
        (review.parent / "renders/render_metrics.json").read_text()
    )
    edit_fractions = [row["edit_pixel_fraction"] for row in render_metrics["views"]]
    comparison["variants"][mode] = {
        **summary["stages"]["refine"],
        "median_diffusion_noise_residual_rms": float(np.median(residuals)),
        "p90_diffusion_noise_residual_rms": float(np.quantile(residuals, 0.9)),
        "mean_dilated_old_new_projection_fraction": float(np.mean(edit_fractions)),
        "max_dilated_old_new_projection_fraction": float(np.max(edit_fractions)),
        "retained_background_bitwise_identical": summary["composition"][
            "retained_background_bitwise_identical"
        ],
        "scene": summary["composition"]["scene"],
        "independent_object": summary["composition"]["object"],
    }
comparison["limits"] = [
    "This tests an unpublished implementation detail, not a different editing method or access to the authors' source.",
    "Reference errors share the same foreground image/mask, but are full-frame and include white background.",
    "Both runs use a shared server and different physical A6000 GPUs; timing differences are not a controlled method-speed comparison.",
    "Background MAE masks are variant-specific old/new projection unions; their values are not directly comparable on an identical pixel set.",
    "Noise residual magnitude is a diagnostic, not a perceptual quality score. Inspect all actual matched views.",
]
(dest / "comparison.json").write_text(json.dumps(comparison, indent=2))
(root / "records/gsle/cfg_comparison.json").write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison["variants"]), flush=True)
