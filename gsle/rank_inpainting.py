"""Automatic semantic ranking of pretrained inpainting candidates, before SAM."""

import json, os, sys, shutil
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image, ImageDraw

root = Path(os.environ["ROOT"])
out = root / "outputs/gsle"
sys.path.insert(0, str(root / "repos/LangSplat"))
import preprocess as official

cfg = json.loads((root / "gsle/config.json").read_text())
pins = json.loads((root / "records/gsle/models.json").read_text())
positives = (
    cfg["replacement_text"],
    "a yellow articulated industrial robot arm on a pedestal",
    "a single six-axis robotic manipulator with an elbow and gripper",
)
negatives = (
    "a yellow roller machine",
    "a long horizontal frame with wheels",
    "two separate industrial robot arms",
    "a disconnected mechanical assembly",
    "texture",
)
clip = official.OpenCLIPNetwork(
    official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(
            Path(pins["openclip"]["path"]) / "open_clip_pytorch_model.bin"
        ),
        positives=positives,
        negatives=negatives,
    )
)
dest = out / "inpainting"
record = json.loads((dest / "generation.json").read_text())
mask = np.asarray(Image.open(out / "avp/bbox_mask.png")) > 127
y, x = np.where(mask)
rows = []
sheet = Image.new("RGB", (512 * len(record["candidates"]), 540), "white")
draw = ImageDraw.Draw(sheet)
for index, candidate in enumerate(record["candidates"]):
    image = np.asarray(Image.open(candidate["path"]).convert("RGB"))
    crop = image.copy()
    crop[~mask] = 0
    crop = crop[y.min() : y.max() + 1, x.min() : x.max() + 1]
    crop = cv2.resize(official.pad_img(crop), (224, 224))
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).cuda().float()[None] / 255
    with torch.no_grad():
        feature = clip.encode_image(tensor)
        feature /= feature.norm(dim=-1, keepdim=True)
        scores = [
            float(clip.get_relevancy(feature, i)[0, 0]) for i in range(len(positives))
        ]
    row = {**candidate, "semantic_scores": scores, "score": float(np.mean(scores))}
    rows.append(row)
    sheet.paste(Image.fromarray(image), (index * 512, 28))
    draw.text(
        (index * 512 + 6, 7),
        f"seed {candidate['seed']} | automatic score {row['score']:.3f}",
        fill="black",
    )
selected = max(rows, key=lambda row: row["score"])
shutil.copyfile(selected["path"], dest / "edited_anchor.png")
record.update(
    selection_pending=False,
    selected=selected,
    candidates=rows,
    selection_rule="mean pretrained OpenCLIP relevance over three robot-arm descriptions against explicit competing machine descriptions; visual QA remains necessary",
    ranking_positive_texts=positives,
    ranking_negative_texts=negatives,
)
(dest / "generation.json").write_text(json.dumps(record, indent=2))
sheet.save(dest / "candidate_sheet.jpg", quality=93)
print("Automatically selected inpainting candidate:", json.dumps(selected), flush=True)
