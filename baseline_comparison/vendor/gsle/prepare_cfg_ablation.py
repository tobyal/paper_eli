"""Isolate the paper's unspecified unconditional-input interpretation."""

import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
source = root / "outputs/gsle"
dest = source / "ablations/text_only_cfg"
records = root / "records/gsle/ablations/text_only_cfg"
dest.mkdir(parents=True, exist_ok=True)
records.mkdir(parents=True, exist_ok=True)
assert not (dest / "refine/final.pth").exists(), "A completed ablation already exists"
for name in ("localization", "avp", "foreground", "coarse", "composition"):
    link = dest / name
    if not link.exists():
        link.symlink_to(source / name, target_is_directory=True)
    assert link.is_symlink() and link.resolve() == (source / name).resolve()
config = json.loads((root / "gsle/config.json").read_text())
assert config["refine"]["unconditional_conditioning"] == "drop_all_eq7"
config["refine"]["unconditional_conditioning"] = "text_only"
(dest / "config.json").write_text(json.dumps(config, indent=2))
inputs = {}
for name in (
    "coarse/final.pth",
    "foreground/reference_white.png",
    "foreground/mask.png",
    "avp/anchor.json",
    "localization/edit_region.json",
    "localization/keep_indices.npy",
):
    digest = hashlib.sha256()
    with (source / name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    inputs[name] = digest.hexdigest()
record = {
    "kind": "Explicit implementation-ambiguity ablation, not a replacement of the primary run",
    "only_config_difference": {
        "refine.unconditional_conditioning": {
            "primary": "drop_all_eq7",
            "ablation": "text_only",
        }
    },
    "shared_input_sha256": inputs,
    "iterations": 2000,
    "seed": config["seed"],
    "meaning": "The ablation retains mask, masked-image context and ControlNet conditions in the empty-text branch, as conventional text CFG does. GSLE Eq7 omits these inputs but does not document nine-channel unconditional padding.",
}
(records / "design.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
