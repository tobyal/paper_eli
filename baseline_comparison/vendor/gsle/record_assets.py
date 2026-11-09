"""Record the already downloaded SAM asset and frozen source identity."""

import hashlib, json, os, subprocess
from pathlib import Path

root = Path(os.environ["ROOT"])
path = root / "models/torch/hub/checkpoints/sam_vit_h_4b8939.pth"
expected = json.loads((root / "records/aux_model_downloads.json").read_text())[
    "sam_vit_h_4b8939.pth"
]
h = hashlib.sha256()
with path.open("rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(block)
assert h.hexdigest() == expected["sha256"]
record = {
    "sam": {"path": str(path), "bytes": path.stat().st_size, **expected},
    "source_identity_record": str(root / "records/common_source_identity.json"),
    "source_ply": str(
        root / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
    ),
    "source_checkpoint": str(root / "outputs/reconstruction/chkpnt30000.pth"),
    "renderer_commit": subprocess.check_output(
        ["git", "-C", str(root / "repos/gaussian-splatting"), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
}
(root / "records/gsle/source_assets.json").write_text(json.dumps(record, indent=2))
print(record, flush=True)
