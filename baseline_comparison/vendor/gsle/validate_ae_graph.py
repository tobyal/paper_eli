"""Compare captured updates with the original eager Adam on actual features."""

import json, os, sys, time
from pathlib import Path
import numpy as np, torch

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat/autoencoder"))
from model import Autoencoder
from ae_graph import AEGraph, ae_loss

torch.manual_seed(42)
data = torch.tensor(
    np.load(root / "data/gsle_langsplat/language_features/frame_000151_f.npy"),
    device="cuda",
)


def model():
    return (
        Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512])
        .cuda()
        .train()
    )


eager = model()
captured = model()
captured.load_state_dict(eager.state_dict())
opt1 = torch.optim.Adam(eager.parameters(), lr=0.0007)
opt2 = torch.optim.Adam(captured.parameters(), lr=0.0007, capturable=True)
runner = AEGraph(captured, opt2, data[:64])
records = []
# Include a remainder between full batches to validate graph/eager state sharing.
for step, size in enumerate([64, 64, 37, 64, 64]):
    batch = data[step * 100 : step * 100 + size]
    opt1.zero_grad(set_to_none=True)
    loss = ae_loss(eager, batch)
    loss.backward()
    opt1.step()
    actual = runner.step(batch)
    torch.cuda.synchronize()
    error = max(
        float((eager.state_dict()[key] - value).abs().max())
        for key, value in captured.state_dict().items()
    )
    assert error < 2e-5, (step, error)
    records.append(
        {
            "update": step + 1,
            "batch": size,
            "maximum_parameter_and_buffer_difference": error,
            "loss_difference": abs(float(loss) - float(actual)),
        }
    )
tick = time.time()
for _ in range(100):
    runner.step(data[:64])
torch.cuda.synchronize()
elapsed = time.time() - tick
(root / "records/gsle/ae_graph_validation.json").write_text(
    json.dumps(
        {
            "actual_features": True,
            "updates": records,
            "100_update_seconds": elapsed,
            "dtype": "float32",
            "optimizer": "Adam, unchanged learning rate/loss; capturable=True for graph execution",
        },
        indent=2,
    )
)
print(records, "100 updates seconds", elapsed, flush=True)
