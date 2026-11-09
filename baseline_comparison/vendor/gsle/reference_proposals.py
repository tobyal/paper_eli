"""Inspect automatic text/instance association at the user's reference view.

These 2D proposals are intermediate diagnostics, NOT the final 3D localization.
The final target must come from the trained multi-scale 3D language fields.
"""

import json, os, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image, ImageDraw

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat"))
import preprocess as official

cfg = json.loads(Path(os.environ["GSLE_CONFIG"]).read_text())
models = json.loads((root / "records/gsle/models.json").read_text())
texts = [cfg["target_text"]]
network = official.OpenCLIPNetwork(
    official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(
            Path(models["openclip"]["path"]) / "open_clip_pytorch_model.bin"
        ),
        positives=tuple(texts),
    )
)
prefix = (
    root / "data/gsle_langsplat/language_features" / Path(cfg["reference_image"]).stem
)
features = torch.as_tensor(
    np.load(str(prefix) + "_f.npy"), device="cuda", dtype=torch.float16
)
segments = np.load(str(prefix) + "_s.npy")
image = np.asarray(
    Image.open(root / "data/industrial/images" / cfg["reference_image"])
    .convert("RGB")
    .resize((938, 528))
)
with torch.no_grad():
    scores = (
        torch.stack(
            [network.get_relevancy(features, i)[:, 0] for i in range(len(texts))], 1
        )
        .cpu()
        .numpy()
    )
records = []
height, width = segments.shape[1:]
for level in (1, 2, 3):
    for index in np.unique(segments[level]):
        if index < 0:
            continue
        mask = segments[level] == index
        y, x = np.where(mask)
        area = len(x) / (height * width)
        if not 0.003 < area < 0.30:
            continue
        cx, cy = float(x.mean() / width), float(y.mean() / height)
        distance = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        relevance = float(scores[index].max())
        score = relevance * np.exp(-4 * distance * distance)
        records.append(
            {
                "level": level,
                "feature_index": int(index),
                "area_fraction": area,
                "centroid": [cx, cy],
                "text_scores": scores[index].tolist(),
                "semantic_score": relevance,
                "rank": float(score),
                "bbox": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            }
        )
records.sort(key=lambda x: x["rank"], reverse=True)
# The user-confirmed original-photo SAM region selects the physical instance;
# its 3D extent is still freshly determined by the LangSplat query and depth.
selected_mask=np.load(cfg["reference_sam_mask"]).astype(bool)
yy,xx=np.where(selected_mask)
chosen={"level":cfg["reference_sam_level"],"feature_index":cfg["reference_sam_ids"][0],
    "feature_indices":cfg["reference_sam_ids"],"area_fraction":float(selected_mask.mean()),
    "centroid":[float(xx.mean()/width),float(yy.mean()/height)],
    "bbox":[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
    "text_scores":scores[cfg["reference_sam_ids"]].mean(0).tolist(),
    "semantic_score":float(scores[cfg["reference_sam_ids"]].mean()),
    "rank":float(scores[cfg["reference_sam_ids"]].mean()),"association":"confirmed original-photo SAM instance"}
records.insert(0,chosen)
dest = Path(os.environ["GSLE_OUTPUT"]) / "localization_proposals"
dest.mkdir(exist_ok=True)
canvas = Image.new("RGB", (468 * 3, 290 * 4), "white")
for rank, item in enumerate(records[:12]):
    mask = segments[item["level"]] == item["feature_index"]
    overlay = image.copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([255, 40, 20])
    thumbnail = Image.fromarray(overlay).resize((468, 264))
    draw = ImageDraw.Draw(thumbnail)
    draw.text(
        (5, 5),
        f"#{rank} level={item['level']} id={item['feature_index']} score={item['rank']:.3f}",
        fill="white",
        stroke_width=1,
        stroke_fill="black",
    )
    canvas.paste(thumbnail, ((rank % 3) * 468, (rank // 3) * 290))
canvas.save(dest / "reference_proposals.jpg")
(dest / "reference_proposals.json").write_text(
    json.dumps(
        {
            "texts": texts,
            "automatic_rank": "max CLIP relevancy * exp(-4 * image-center-distance^2)",
            "proposals": records,
            "not_final_3d_localization": True,
        },
        indent=2,
    )
)
print(records[:12], flush=True)
