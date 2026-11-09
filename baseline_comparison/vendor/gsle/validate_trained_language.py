"""Check actual trained fields against the immutable reconstruction checkpoint."""

import argparse, json, os, time
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, default=7000)
args = parser.parse_args()
root = Path(os.environ["ROOT"])
start = time.time()
source, _ = torch.load(
    root / "outputs/reconstruction/chkpnt30000.pth", map_location="cpu"
)
rows = []
for level in (1, 2, 3):
    model, iteration = torch.load(
        root / f"outputs/gsle/langsplat_{level}/chkpnt{args.iteration}.pth",
        map_location="cpu",
    )
    assert iteration == args.iteration and len(model) == 13
    for index in range(1, 7):
        assert torch.equal(source[index], model[index]), (level, index)
    field = model[7].detach()
    assert torch.isfinite(field).all()
    norms = field.norm(dim=-1)
    rows.append(
        {
            "level": level,
            "iteration": iteration,
            "source_geometry_and_appearance_exactly_identical": True,
            "gaussians": len(field),
            "finite_language_features": True,
            "nonzero_language_fraction": float((norms > 0).float().mean()),
            "language_norm_mean": float(norms.mean()),
        }
    )
    del model, field
record = {
    "levels": rows,
    "seconds": time.time() - start,
    "scope": "Actual saved training checkpoints, not synthetic probe",
}
(root / f"records/gsle/language_parameter_identity_{args.iteration}.json").write_text(
    json.dumps(record, indent=2)
)
print(json.dumps(record), flush=True)
