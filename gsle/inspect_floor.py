"""Inspect automatic floor evidence without editing Gaussian parameters."""

from common import *
from world_up import estimate_up

source = GaussianData.load(
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply",
    device="cpu",
)
xyz = source.xyz.numpy()
scales = source.log_scale.exp().numpy()
opacity = source.opacity_logit.sigmoid().numpy()[:, 0]
scores = np.max(
    [
        np.load(OUT / f"localization/language_query_level{level}.npz")["floor_scores"]
        for level in (1, 2, 3)
    ],
    axis=0,
)
cameras = source_cameras()
up, record = estimate_up(xyz, scales, opacity, scores, cameras)
selected = (scores > 0.55) & (opacity > 0.1) & (scales.min(1) / scales.max(1) < 0.3)
heights = xyz[selected] @ up
record["floor_candidate_height_quantiles"] = np.quantile(
    heights, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
).tolist()
record["camera_height_quantiles"] = np.quantile(
    np.array([c.center @ up for c in cameras]), [0, 0.5, 1]
).tolist()
counts, edges = np.histogram(heights, bins=100)
record["height_histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
save_json(OUT / "localization/floor_diagnostics.json", record)
print(json.dumps(record), flush=True)
