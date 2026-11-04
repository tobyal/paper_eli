"""Depth-conditioned anchor inpainting; save real frozen-model output."""

from PIL import Image
from common import *
from diffusion import DepthInpaintingGuidance

cfg = config()
dest = OUT / "inpainting"
dest.mkdir(exist_ok=True)
source = OUT / "avp"
model = DepthInpaintingGuidance()
seeds = cfg["inpainting"].get("candidate_seeds", [cfg["inpainting"]["seed"]])
(dest / "candidates").mkdir(exist_ok=True)
candidates = []
for seed in seeds:
    result = model.inpaint(
        Image.open(source / "anchor.png").convert("RGB"),
        Image.open(source / "bbox_mask.png").convert("L"),
        Image.open(source / "depth_control.png").convert("RGB"),
        cfg["replacement_text"],
        seed,
    )
    path = dest / f"candidates/seed_{seed}.png"
    result.save(path)
    candidates.append({"seed": seed, "path": str(path)})
    print("Generated depth-conditioned candidate", seed, flush=True)
if len(candidates) == 1:
    result.save(dest / "edited_anchor.png")
save_json(
    dest / "generation.json",
    {
        "text": cfg["replacement_text"],
        "settings": cfg["inpainting"],
        "anchor": str(source / "anchor.json"),
        "weights": json.loads((ROOT / "records/gsle/models.json").read_text()),
        "output": str(dest / "edited_anchor.png"),
        "candidates": candidates,
        "selection_pending": len(candidates) > 1,
    },
)
print("Depth-conditioned candidates saved:", dest / "candidates", flush=True)
