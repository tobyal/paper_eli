"""M1: reuse verified LangSplat localization without retraining or changing it."""
import shutil
import numpy as np
from ..core.context import Context, arguments, read_json, write_json, sha256
from ..core.scene import partition
from ..core.cameras import region_frame


def run(ctx):
    out = ctx.stage("m1")
    source = ctx.source(ctx.cfg["project"]["source_ply"])
    assert sha256(source) == ctx.cfg["project"]["source_sha256"], "Source scene changed"
    loc = ctx.source(ctx.cfg["m1"]["localization_dir"])
    target = read_json(loc / "target.json")
    region = read_json(loc / "edit_region.json")
    reference = read_json(ctx.source(ctx.cfg["m2"]["existing_anchor"]))["camera"]
    for name in ("target.json", "edit_region.json", "target_indices.npy", "keep_indices.npy"):
        shutil.copy2(loc / name, out / name)
    ids, keep = np.load(out / "target_indices.npy"), np.load(out / "keep_indices.npy")
    count = partition(source, ids, keep, out)
    manifest = {"source_scene": str(source), "source_sha256": sha256(source),
                "target_gaussians": str(out / "target_gaussians.ply"),
                "background_gaussians": str(out / "background_gaussians.ply"),
                "target_bbox": target["bbox_corners"], "generation_bbox": region["bbox_corners"],
                "region_center": region["center"], "world_up": region["up"],
                "reference_camera": reference, "localization_camera": target["reference_camera"],
                "world_transform": region_frame(region, reference),
                "counts": {"source": count, "target": len(ids), "background": len(keep)},
                "localization_reused": True, "manual_roi_used": False}
    write_json(out / "manifest.json", manifest)
    ctx.complete("m1", [out / n for n in ("manifest.json", "target_gaussians.ply", "background_gaussians.ply", "target_indices.npy", "keep_indices.npy")], counts=manifest["counts"])
    print("M1 complete", manifest["counts"], flush=True)


if __name__ == "__main__":
    run(Context(arguments().config))
