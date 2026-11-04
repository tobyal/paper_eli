"""Image-derived SAM prompts for the user's explicitly yellow replacement.

Color components supply points only. SAM predicts every final mask pixel; no
color mask is substituted for its segmentation and no coordinate is hand set.
"""

import cv2, numpy as np


def yellow_object_prompts(image, rough_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = (
        rough_mask
        & (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 40)
        & (hsv[:, :, 1] >= 100)
        & (hsv[:, :, 2] >= 70)
    )
    gray = rough_mask & (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 90)
    points = []
    labels = []
    components = []
    for name, binary, label, minimum, maximum in [
        ("yellow", yellow, 1, 80, 8),
        ("gray_background", gray, 0, max(1000, int(rough_mask.sum() * 0.15)), 3),
    ]:
        count, ids, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        order = sorted(
            range(1, count),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        for index in order[:maximum]:
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < minimum:
                continue
            distance = cv2.distanceTransform(
                (ids == index).astype(np.uint8), cv2.DIST_L2, 5
            )
            point = cv2.minMaxLoc(distance)[3]
            points.append(point)
            labels.append(label)
            components.append(
                {"kind": name, "area": area, "point": list(point), "label": label}
            )
    if not any(labels):
        raise RuntimeError("No automatic yellow-object seed found")
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        {
            "method": "interior distance-transform peaks of yellow components and large low-saturation components inside preliminary SAM mask",
            "components": components,
            "manual_coordinates": False,
            "final_mask_source": "SAM, not HSV thresholding",
        },
    )
