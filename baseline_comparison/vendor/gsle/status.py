"""Compact, artifact-based status; never infer completed editing from code files."""

import json, os, re, time
from pathlib import Path

root = Path(os.environ["ROOT"])
out = root / "outputs/gsle"
records = root / "records/gsle"
status = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "features": {
        "completed": len(
            list((root / "data/gsle_langsplat/language_features").glob("*_f.npy"))
        ),
        "total": 306,
    },
    "language_fields": {},
}
if (records / "autoencoder_progress.json").exists():
    status["autoencoder"] = json.loads(
        (records / "autoencoder_progress.json").read_text()
    )
for level in (1, 2, 3):
    path = root / f"logs/gsle/04_language_level{level}.log"
    progress = 0
    if path.exists():
        with path.open("rb") as f:
            f.seek(max(path.stat().st_size - 10000, 0))
            tail = f.read().decode(errors="replace")
        matches = re.findall(r"(\d+)/30000", tail)
        if matches:
            progress = int(matches[-1])
    status["language_fields"][level] = {
        "iteration": progress,
        "checkpoint_complete": (out / f"langsplat_{level}/chkpnt30000.pth").exists(),
    }
for stage, marker in {
    "localization": "localization/target.json",
    "avp": "avp/anchor.json",
    "inpainting": "inpainting/edited_anchor.png",
    "sam": "foreground/object_rgba.png",
    "coarse": "coarse/final_summary.json",
    "composition": "composition/composition.json",
    "refine": "refine/final_summary.json",
    "final_scene": "final/composition.json",
    "final_video": "final/renders/comparison_original_speed_2fps.mp4",
}.items():
    status[stage] = {"artifact_exists": (out / marker).exists()}
for stage in ("coarse", "refine"):
    if (records / f"{stage}_progress.json").exists():
        status[stage]["progress"] = json.loads(
            (records / f"{stage}_progress.json").read_text()
        )
alternative = out / "ablations/text_only_cfg"
if alternative.exists():
    status["cfg_ablation"] = {
        "mode": "text_only",
        "refinement_complete": (alternative / "refine/final_summary.json").exists(),
        "final_scene_exists": (alternative / "final/composition.json").exists(),
        "final_video_exists": (
            alternative / "final/renders/comparison_original_speed_2fps.mp4"
        ).exists(),
    }
    progress = records / "ablations/text_only_cfg/refine_progress.json"
    if progress.exists():
        status["cfg_ablation"]["progress"] = json.loads(progress.read_text())
(records / "current_status.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status, ensure_ascii=False))
