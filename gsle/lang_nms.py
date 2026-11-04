"""Exact vectorization of LangSplat mask NMS; thresholds/order unchanged.

All intersections are integer counts representable exactly in float32.
Chunked binary GEMM replaces the original nested HxW logical-and/or loops.
"""

import torch


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    scores, idx = scores.sort(0, descending=True)
    n = len(idx)
    ordered = masks[idx].flatten(1)
    counts = torch.zeros(n, n, device="cuda", dtype=torch.float32)
    area = ordered.sum(1, dtype=torch.float32).cuda()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for chunk in ordered.split(16384, dim=1):
            x = chunk.to(device="cuda", dtype=torch.float32)
            counts.add_(x @ x.T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    union = area[:, None] + area[None, :] - counts
    iou = counts / union
    ai = counts / area[:, None]
    aj = counts / area[None, :]
    upper = torch.triu(torch.ones(n, n, device="cuda", dtype=torch.bool))
    inner = torch.zeros_like(counts)
    condition = (ai < 0.5) & (aj >= 0.85) & upper
    inner[condition] = 1 - ai[condition] * aj[condition]
    condition = (ai >= 0.85) & (aj < 0.5) & upper
    values = torch.where(condition, 1 - ai * aj, 0).T
    inner = torch.maximum(inner, values)
    keep = torch.triu(iou, diagonal=1).max(0).values <= iou_thr
    conf = scores.to("cuda") > score_thr
    inner_u = torch.triu(inner, diagonal=1).max(0).values <= 1 - inner_thr
    inner_l = torch.tril(inner, diagonal=1).max(0).values <= 1 - inner_thr
    for flag in (conf, inner_u, inner_l):
        if flag.sum() == 0:
            # Upstream uses [index,0] on a 1-D tensor and topk(3) for n<3.
            # Preserve the stated top-three fallback while fixing the indexing.
            flag[scores.to("cuda").topk(min(3, n)).indices] = True
    return idx[(keep & conf & inner_u & inner_l).cpu()]


def verify_against_official(official, masks, scores):
    masks = masks[:32]
    scores = scores[:32]
    options = {"iou_thr": 0.8, "score_thr": 0.7, "inner_thr": 0.5}
    expected = official(masks, scores, **options)
    actual = mask_nms(masks, scores, **options)
    assert torch.equal(expected, actual), (expected, actual)
    return {
        "actual_masks_compared": len(masks),
        "selected_indices": actual.tolist(),
        "exact_agreement": True,
    }
