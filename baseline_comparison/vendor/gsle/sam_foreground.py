"""Automatically prompt SAM using the projected 3D editing region.

No hand-drawn mask or manually provided object image. All masks and CLIP
ranking scores are retained so failures can be inspected before 3D training.
Run in gsle_langsplat (the pinned SAM/OpenCLIP environment).
"""

import json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

root = Path(os.environ["ROOT"])
out = Path(os.environ["GSLE_OUTPUT"])
sys.path.insert(0, str(root / "repos/LangSplat"))
import preprocess as official
from segment_anything import SamPredictor


cfg = json.loads(Path(os.environ["GSLE_CONFIG"]).read_text())
pins = json.loads((root / "records/gsle/models.json").read_text())
image = np.asarray(Image.open(out / "inpainting/edited_anchor.png").convert("RGB"))
edit_mask = np.asarray(Image.open(out / "avp/bbox_mask.png")) > 127
y, x = np.where(edit_mask)
box = np.array([x.min(), y.min(), x.max(), y.max()])
sam = official.sam_model_registry["vit_h"](
    checkpoint=str(root / "models/torch/hub/checkpoints/sam_vit_h_4b8939.pth")
).cuda()
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, logits = predictor.predict(box=box, multimask_output=True)
points=np.empty((0,2),np.float32);labels=np.empty((0,),np.int32)
prompt_record={"method":"projected editing-region box and exact replacement CLIP text", "box":box.tolist(),"color_rules":None}
network = official.OpenCLIPNetwork(
    official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(
            Path(pins["openclip"]["path"]) / "open_clip_pytorch_model.bin"
        ),
        positives=(cfg["replacement_text"],),
    )
)
dest = out / "foreground"
dest.mkdir(exist_ok=True)
(dest / "automatic_prompts.json").write_text(json.dumps(prompt_record, indent=2))
records = []
for i, (mask, sam_score) in enumerate(zip(masks, scores)):
    sy, sx = np.where(mask)
    if len(sx) < 20:
        continue
    crop = image.copy()
    crop[~mask] = 0
    crop = crop[sy.min() : sy.max() + 1, sx.min() : sx.max() + 1]
    crop = cv2.resize(official.pad_img(crop), (224, 224))
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).cuda().float()[None] / 255
    with torch.no_grad():
        embed = network.encode_image(tensor)
        embed /= embed.norm(dim=-1, keepdim=True)
        relevance = float(network.get_relevancy(embed, 0)[0, 0])
    containment = float((mask & edit_mask).sum() / mask.sum())
    rank = relevance * float(sam_score) * containment
    sampled = mask[points[:, 1].astype(int), points[:, 0].astype(int)]
    positive_coverage = float(sampled[labels == 1].mean()) if (labels == 1).any() else 1.0
    negative_coverage = (
        float(sampled[labels == 0].mean()) if (labels == 0).any() else 0.0
    )
    valid = containment > 0.7 and not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    Image.fromarray(mask.astype(np.uint8) * 255).save(dest / f"candidate_{i}.png")
    records.append(
        {
            "index": i,
            "sam_iou": float(sam_score),
            "clip_relevance": relevance,
            "containment": containment,
            "area": int(mask.sum()),
            "score": rank,
            "positive_prompt_coverage": positive_coverage,
            "negative_prompt_coverage": negative_coverage,
            "valid": valid,
        }
    )
assert records, "SAM produced no valid foreground"
valid_records = [row for row in records if row["valid"]]
assert (
    valid_records
), "SAM did not satisfy automatic foreground/background prompts; inspect saved candidates"
selected = max(valid_records, key=lambda r: r["score"])
mask = masks[selected["index"]]
Image.fromarray(mask.astype(np.uint8) * 255).save(dest / "mask.png")
Image.fromarray(
    np.concatenate((image, mask[:, :, None].astype(np.uint8) * 255), axis=-1)
).save(dest / "object_rgba.png")
white = image.copy()
white[~mask] = 255
Image.fromarray(white).save(dest / "reference_white.png")
(dest / "segmentation.json").write_text(
    json.dumps(
        {
            "prompt_source": "projected3D editing box plus SAM multimask and exact replacement-text OpenCLIP ranking; no color-specific rules",
            "automatic_points": prompt_record,
            "box": box.tolist(),
            "candidates": records,
            "selected": selected,
            "manual_prompt": False,
        },
        indent=2,
    )
)
print("SAM selected", selected, flush=True)
