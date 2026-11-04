"""Parallel execution of SAM's unchanged region cleanup and box NMS.

Adapted from Meta SAM, Apache-2.0; see LICENSE.SAM. Work is dispatched in
input order, so proposal ranking and ties remain identical to upstream.
"""

from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision.ops.boxes import batched_nms
from segment_anything.utils.amg import (
    rle_to_mask,
    remove_small_regions,
    batched_mask_to_box,
    mask_to_rle_pytorch,
)

pool = ThreadPoolExecutor(max_workers=8)


def postprocess_small_regions(mask_data, min_area, nms_thresh):
    if not len(mask_data["rles"]):
        return mask_data

    def clean(rle):
        mask = rle_to_mask(rle)
        mask, holes_changed = remove_small_regions(mask, min_area, mode="holes")
        mask, islands_changed = remove_small_regions(mask, min_area, mode="islands")
        return torch.as_tensor(mask)[None], float(
            not holes_changed and not islands_changed
        )

    cleaned = list(pool.map(clean, mask_data["rles"]))
    masks = torch.cat([x[0] for x in cleaned])
    scores = [x[1] for x in cleaned]
    boxes = batched_mask_to_box(masks)
    keep = batched_nms(
        boxes.float(),
        torch.tensor(scores),
        torch.zeros_like(boxes[:, 0]),
        iou_threshold=nms_thresh,
    )
    for idx in keep:
        if scores[idx] == 0:
            mask_data["rles"][idx] = mask_to_rle_pytorch(masks[idx][None])[0]
            mask_data["boxes"][idx] = boxes[idx]
    mask_data.filter(keep)
    return mask_data
