"""LangSplat's exact AE architecture/loss/100-epoch schedule with resumable I/O.

Official model.py and dataset.py are imported unchanged. The feature tensor is
cached on GPU, and per-minibatch histogram/anomaly debugging is omitted. These
execution adaptations do not change the optimizer, batch size or losses.
"""

import json, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat/autoencoder"))
from dataset import Autoencoder_dataset
from model import Autoencoder
from ae_graph import AEGraph
from provenance import capture
from validate_feature_cache import validate

capture("autoencoder")
torch.manual_seed(42)
np.random.seed(42)
dataset = root / "data/gsle_langsplat"
dest = root / "repos/LangSplat/autoencoder/ckpt/industrial_gsle"
dest.mkdir(parents=True, exist_ok=True)
assert len(list((dataset / "language_features").glob("*_f.npy"))) == 306
validate()
source = Autoencoder_dataset(str(dataset / "language_features"))
# Parallel feature extraction changes directory insertion order. Fix filename
# order before seeded shuffling so scheduling cannot change the AE dataset.
offset = 0
slices = {}
for name, count in source.data_dic.items():
    slices[name] = source.data[offset : offset + count]
    offset += count
names = sorted(source.data_dic)
source.data = np.concatenate([slices[name] for name in names])
source.data_dic = {name: source.data_dic[name] for name in names}
data = torch.as_tensor(source.data, device="cuda")
assert (
    len(data) % 64 != 1
), "Singleton final batch needs an explicit BatchNorm compatibility handling"
model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).cuda()
assert (
    root / "records/gsle/ae_graph_validation.json"
).exists(), "Run gsle/validate_ae_graph.py first"
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, capturable=True)
first = 0
best = 100.0
best_epoch = 0
last = dest / "resume.pth"
if last.exists():
    state = torch.load(last)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    first = state["epoch"] + 1
    best = state["best"]
    best_epoch = state["best_epoch"]
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def losses(out, gt):
    return (out - gt).square().mean(), 1 - F.cosine_similarity(out, gt, dim=0).mean()


runner = AEGraph(model, optimizer, data[:64])
start = time.time()
for epoch in range(first, 100):
    tick = time.time()
    model.train()
    order = torch.randperm(len(data), device="cpu").cuda()
    total = torch.zeros((), device="cuda")
    for index in order.split(64):
        batch = data[index]
        loss = runner.step(batch)
        total.add_(loss * len(batch))
    evaluation = None
    if epoch > 95:
        model.eval()
        evaluation = 0.0
        with torch.no_grad():
            for batch in data.split(256):
                l2, cos = losses(model(batch), batch)
                evaluation += float(l2 + cos) * len(batch)
        evaluation /= len(data)
        if evaluation < best:
            best = evaluation
            best_epoch = epoch
            torch.save(model.state_dict(), dest / "best_ckpt.pth")
    record = {
        "epoch": epoch,
        "train_loss": float(total) / len(data),
        "evaluation_loss": evaluation,
        "seconds": time.time() - tick,
        "elapsed": time.time() - start,
        "feature_vectors": len(data),
        "best_epoch": best_epoch,
        "best_loss": best,
    }
    print(record, flush=True)
    with (root / "logs/gsle/autoencoder_epochs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    (root / "records/gsle/autoencoder_progress.json").write_text(
        json.dumps(record, indent=2)
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        last,
    )
model.load_state_dict(torch.load(dest / "best_ckpt.pth"))
model.eval()
compressed = []
with torch.no_grad():
    for batch in data.split(256):
        compressed.append(model.encode(batch).cpu().numpy())
compressed = np.concatenate(compressed)
output = dataset / "language_features_dim3"
output.mkdir(exist_ok=True)
offset = 0
for name, n in source.data_dic.items():
    np.save(output / (name + ".npy"), compressed[offset : offset + n])
    offset += n
    segmentation = name[:-2] + "_s.npy"
    link = output / segmentation
    if not link.exists():
        link.symlink_to(dataset / "language_features" / segmentation)
(root / "records/gsle/autoencoder.json").write_text(
    json.dumps(
        {
            "epochs": 100,
            "batch": 64,
            "lr": 0.0007,
            "encoder": [256, 128, 64, 32, 3],
            "decoder": [16, 32, 64, 128, 256, 256, 512],
            "training_loss": "MSE + .001*cosine(dim=0)",
            "checkpoint_metric": "MSE + cosine(dim=0), epochs96..99",
            "best_epoch": best_epoch,
            "best_loss": best,
            "vectors": len(data),
            "seed": 42,
            "execution_adaptations": [
                "GPU feature cache",
                "no per-batch histograms",
                "no anomaly-debug tracing",
                "epoch checkpoints",
                "validated CUDA graph with capturable Adam; fp32 equations unchanged; numerical comparison in ae_graph_validation.json",
            ],
        },
        indent=2,
    )
)
print("Autoencoder and dimension-3 features completed", flush=True)
