"""Reusable SAM/OpenCLIP adapter with automatic prompts for the yellow target."""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from .context import Context, arguments, read_json, write_json


def run(ctx):
    out = ctx.stage("m2")
    sys.path.insert(0, str(ctx.root / "repos/LangSplat"))
    sys.path.insert(0, str(ctx.root / "gsle"))
    import preprocess as official
    from segment_anything import SamPredictor
    image = np.asarray(Image.open(out / "edited_scene.png").convert("RGB"))
    edit = np.asarray(Image.open(out / "edit_mask.png").resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)) > 127
    y, x = np.where(edit)
    box = np.array([x.min(), y.min(), x.max(), y.max()])
    sam = official.sam_model_registry["vit_h"](checkpoint=str(ctx.root / "models/torch/hub/checkpoints/sam_vit_h_4b8939.pth")).cuda()
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    mode=ctx.cfg['m2'].get('segmentation_prompt_mode','yellow_robot')
    if mode=='box_clip':
        points=np.empty((0,2),np.float32);labels=np.empty(0,np.int32)
        prompt_record={'method':'automatic_edit_volume_box_and_replacement_text','box':box.tolist(),
            'color_rule':None,'text':ctx.cfg['m2']['replacement_text']}
    elif mode=='yellow_robot':
        from sam_prompts import yellow_object_prompts
        points, labels, prompt_record = yellow_object_prompts(image, masks[int(scores.argmax())])
        refined, refined_scores, _ = predictor.predict(box=box, point_coords=points, point_labels=labels, multimask_output=True)
        masks, scores = np.concatenate((masks, refined)), np.concatenate((scores, refined_scores))
    else:
        raise ValueError('Unknown foreground segmentation prompt mode: '+mode)
    pins = read_json(ctx.root / "records/gsle/models.json")
    network = official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(clip_model_pretrained=str(Path(pins["openclip"]["path"]) / "open_clip_pytorch_model.bin"), positives=(ctx.cfg["m2"]["replacement_text"],)))
    candidates = []
    for i, (mask, confidence) in enumerate(zip(masks, scores)):
        yy, xx = np.where(mask)
        if len(xx) < 20:
            continue
        crop = image.copy()
        crop[~mask] = 0
        crop = cv2.resize(official.pad_img(crop[yy.min():yy.max()+1, xx.min():xx.max()+1]), (224, 224))
        with torch.no_grad():
            embedding = network.encode_image(torch.from_numpy(crop.copy()).permute(2, 0, 1).cuda().float()[None] / 255)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            relevance = float(network.get_relevancy(embedding, 0)[0, 0])
        covered = mask[points[:, 1].astype(int), points[:, 0].astype(int)]
        pos = float(covered[labels == 1].mean()) if (labels == 1).any() else 1.
        neg = float(covered[labels == 0].mean()) if (labels == 0).any() else 0
        containment = float((mask & edit).sum() / mask.sum())
        clipped = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        candidates.append({"index": i, "sam_score": float(confidence), "clip_score": relevance,
                           "positive_coverage": pos, "negative_coverage": neg, "containment": containment,
                           "image_border_clipped": clipped, "area": int(mask.sum()),
                           "valid": pos >= 0.9 and neg == 0 and not clipped and containment > 0.7,
                           "score": relevance * float(confidence) * containment})
        directory = out / "sam_candidates"
        directory.mkdir(exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"candidate_{i}.png")
    valid = [v for v in candidates if v["valid"]]
    write_json(out / "automatic_prompts.json", prompt_record)
    write_json(out / "segmentation.json", {"candidates": candidates, "manual_prompt": False})
    if not valid:
        raise RuntimeError("SAM candidates failed foreground/background/clipping checks; inspect saved masks. Qwen will not be called again.")
    chosen = max(valid, key=lambda value: value["score"])
    mask = masks[chosen["index"]]
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "object_mask.png")
    Image.fromarray(np.concatenate((image, mask[:, :, None].astype(np.uint8) * 255), -1)).save(out / "object_rgba.png")
    white = image.copy()
    white[~mask] = 255
    Image.fromarray(white).save(out / "object_white.png")
    write_json(out / "segmentation.json", {"candidates": candidates, "selected": chosen, "automatic_prompts": prompt_record, "manual_prompt": False})
    print("SAM selected", chosen, flush=True)


if __name__ == "__main__":
    run(Context(arguments().config))
