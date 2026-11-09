"""Resumable I/O adapter around official LangSplat multiscale SAM/CLIP code."""

import argparse, json, os, sys, time, faulthandler
from pathlib import Path
import cv2, numpy as np, torch

cv2.setNumThreads(1)
root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat"))
import preprocess as official
from lang_nms import mask_nms, verify_against_official
from sam_regions import postprocess_small_regions
from copy import deepcopy

faulthandler.dump_traceback_later(180, repeat=True)
original_nms = official.mask_nms
validated = False


def logged_nms(masks, scores, **kwargs):
    global validated
    print("NMS input", list(masks.shape), flush=True)
    if not validated:
        check = verify_against_official(original_nms, masks, scores)
        (root / "records/gsle/nms_equivalence.json").write_text(
            json.dumps(check, indent=2)
        )
        validated = True
    tick = time.time()
    result = mask_nms(masks, scores, **kwargs)
    print("NMS complete", len(result), "seconds", time.time() - tick, flush=True)
    return result


official.mask_nms = logged_nms
original_crop = official.get_seg_img
crop_checks = 0


def crop_before_mask(mask, image):
    global crop_checks
    x, y, w, h = np.int32(mask["bbox"])
    cropped = image[y : y + h, x : x + w].copy()
    cropped[mask["segmentation"][y : y + h, x : x + w] == 0] = 0
    if crop_checks < 32:
        assert np.array_equal(cropped, original_crop(mask, image))
        crop_checks += 1
        if crop_checks == 32:
            (root / f"records/gsle/crop_equivalence_{args.shard}.json").write_text(
                json.dumps(
                    {"actual_masks_compared": 32, "exact_rgb_crop_agreement": True},
                    indent=2,
                )
            )
    return cropped


official.get_seg_img = crop_before_mask
p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=0)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--shards", type=int, default=1)
p.add_argument("--only-image")
args = p.parse_args()
official.seed_everything(42)
models = json.loads((root / "records/gsle/models.json").read_text())
config = official.OpenCLIPNetworkConfig(
    clip_model_pretrained=str(
        Path(models["openclip"]["path"]) / "open_clip_pytorch_model.bin"
    )
)
official.model = official.OpenCLIPNetwork(config)
sam = official.sam_model_registry["vit_h"](
    checkpoint=str(root / "models/torch/hub/checkpoints/sam_vit_h_4b8939.pth")
).cuda()
official.mask_generator = official.SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)
original_cleanup = official.mask_generator.postprocess_small_regions
cleanup_validated = False


def checked_cleanup(data, min_area, nms_thresh):
    global cleanup_validated
    if not cleanup_validated:
        subset = deepcopy(data)
        subset.filter(torch.arange(min(16, len(data["rles"]))))
        expected = original_cleanup(deepcopy(subset), min_area, nms_thresh)
        actual = postprocess_small_regions(deepcopy(subset), min_area, nms_thresh)
        assert expected["rles"] == actual["rles"]
        assert np.array_equal(expected["boxes"], actual["boxes"])
        (root / f"records/gsle/sam_cleanup_equivalence_{args.shard}.json").write_text(
            json.dumps(
                {
                    "mask_count": len(subset["rles"]),
                    "exact_mask_and_box_agreement": True,
                    "workers": 8,
                },
                indent=2,
            )
        )
        cleanup_validated = True
    return postprocess_small_regions(data, min_area, nms_thresh)


official.mask_generator.postprocess_small_regions = checked_cleanup
original_generate = official.mask_generator.generate


def timed_generate(image):
    tick = time.time()
    result = original_generate(image)
    print(
        "SAM complete",
        list(map(len, result)),
        "seconds",
        time.time() - tick,
        flush=True,
    )
    return result


official.mask_generator.generate = timed_generate
dataset = root / "data/gsle_langsplat"
dest = dataset / "language_features"
dest.mkdir(exist_ok=True)
start = time.time()
done = 0
paths = sorted((dataset / "images").iterdir())
if args.only_image:
    paths = [path for path in paths if path.name == args.only_image]
paths = [path for i, path in enumerate(paths) if i % args.shards == args.shard]
paths.sort(key=lambda path: (path.name != "frame_000151.jpg", path.name))
for path in paths:
    prefix = dest / path.stem
    if Path(str(prefix) + "_f.npy").exists() and Path(str(prefix) + "_s.npy").exists():
        continue
    image = cv2.resize(cv2.imread(str(path)), (938, 528))
    print("Begin SAM/CLIP", path.name, flush=True)
    tick = time.time()
    with torch.no_grad():
        embeds, maps = official._embed_clip_sam_tiles(
            torch.from_numpy(image).permute(2, 0, 1)[None], official.sam_encoder
        )
    lengths = [len(v) for v in embeds.values()]
    offset = 0
    segments = []
    for level, (key, values) in enumerate(maps.items()):
        assert values.max() == lengths[level] - 1, (path, key)
        values[values != -1] += offset
        segments.append(values)
        offset += lengths[level]
    feature = torch.cat(list(embeds.values())).float().numpy()
    segment = np.stack(segments).astype(np.int32)
    assert segment.shape == (4, 528, 938) and offset == len(feature)
    # Feature file written last is the per-image completion marker.
    np.save(str(prefix) + "_s.npy", segment)
    np.save(str(prefix) + "_f.npy", feature)
    done += 1
    progress = {
        "last_image": path.name,
        "feature_count": len(feature),
        "seconds": time.time() - tick,
        "completed": len(list(dest.glob("*_f.npy"))),
        "total": 306,
        "elapsed_this_run": time.time() - start,
    }
    (root / "records/gsle/language_feature_progress.json").write_text(
        json.dumps(progress, indent=2)
    )
    print(progress, flush=True)
    if args.limit and done >= args.limit:
        break
