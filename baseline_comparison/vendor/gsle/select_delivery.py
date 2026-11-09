"""Index reviewed, verified results without modifying either experimental run."""

import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=("drop_all_eq7", "text_only"))
args = parser.parse_args()
root = Path(os.environ["ROOT"])
out = root / "outputs/gsle"
review = json.loads((root / "records/gsle/final_visual_review.json").read_text())
assert review["selected_variant"] == args.variant
base = out if args.variant == "drop_all_eq7" else out / "ablations/text_only_cfg"
records = (
    root / "records/gsle"
    if args.variant == "drop_all_eq7"
    else root / "records/gsle/ablations/text_only_cfg"
)
assert (records / "final_artifact_verification.json").is_file()
summary = json.loads((base / "final/review/summary.json").read_text())
assert summary["stages"]["coarse"]["optimizer_updates"] == 6500
assert summary["stages"]["refine"]["optimizer_updates"] == 2000
files = {
    "scene.ply": base / "final/scene.ply",
    "object_world.ply": base / "final/object_world.ply",
    "object_local.ply": base / "refine/final_object_local.ply",
    "object_checkpoint.pth": base / "refine/final.pth",
    "effective_config.json": base / "refine/effective_config.json",
    "render_metrics_and_cameras.json": base / "final/renders/render_metrics.json",
    "run_summary.json": base / "final/review/summary.json",
    "anchor_comparison.png": base / "final/review/anchor_comparison.png",
    "matched_views_1.jpg": base / "final/review/matched_views_1.jpg",
    "matched_views_2.jpg": base / "final/review/matched_views_2.jpg",
    "training_curves.png": base / "final/review/training_curves.png",
    "comparison_original_speed_2fps.mp4": base
    / "final/renders/comparison_original_speed_2fps.mp4",
    "comparison_6x_preview_12fps.mp4": base
    / "final/renders/comparison_6x_preview_12fps.mp4",
    "edited_original_speed_2fps.mp4": base
    / "final/renders/edited_original_speed_2fps.mp4",
    "object_360.mp4": base / "final/renders/object_360.mp4",
    "scene_360.mp4": base / "final/renders/scene_360.mp4",
    "cfg_comparison.png": out / "ablation_comparison/anchor_comparison.png",
}
artifacts = []
for filename, path in files.items():
    assert path.is_file(), path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    artifacts.append(
        {
            "filename": filename,
            "server_path": str(path),
            "archive_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "preview": path.suffix in (".png", ".jpg", ".mp4"),
        }
    )
record = {
    "selected_variant": args.variant,
    "server_root": str(root),
    "source_video_seconds": 174.89655,
    "comparison_video_seconds": 175.0,
    "preview_speed_multiplier": 6,
    "visual_review": review,
    "artifacts": artifacts,
    "other_experiments_preserved": True,
    "complete_module_artifacts_bundle": "gsle_results.tar.gz",
}
(root / "records/gsle/deliverables.json").write_text(json.dumps(record, indent=2))
print(
    json.dumps(
        {
            "selected_variant": args.variant,
            "files": len(artifacts),
            "bytes": sum(item["bytes"] for item in artifacts),
        }
    ),
    flush=True,
)
