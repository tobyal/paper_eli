"""Decode all three trained LangSplat fields and evaluate text in 3D."""

import json, os, sys, time
from pathlib import Path
import numpy as np, torch

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "repos/LangSplat"))
import preprocess as official

sys.path.insert(0, str(root / "repos/LangSplat/autoencoder"))
from model import Autoencoder

out = Path(os.environ["GSLE_OUTPUT"])
dest = out / "localization"
dest.mkdir(exist_ok=True)
proposals = json.loads(
    (out / "localization_proposals/reference_proposals.json").read_text()
)
texts = proposals["texts"]
selected = proposals["proposals"][0]
pins = json.loads((root / "records/gsle/models.json").read_text())
clip = official.OpenCLIPNetwork(
    official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(
            Path(pins["openclip"]["path"]) / "open_clip_pytorch_model.bin"
        ),
        positives=tuple(texts + ["factory floor"]),
    )
)
decoder = (
    Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).cuda().eval()
)
decoder.load_state_dict(
    torch.load(root / "repos/LangSplat/autoencoder/ckpt/industrial_gsle/best_ckpt.pth")
)
source, _ = torch.load(
    root / "outputs/reconstruction/chkpnt30000.pth", map_location="cpu"
)
cfg = json.loads(Path(os.environ["GSLE_CONFIG"]).read_text())
prefix = (
    root / "data/gsle_langsplat/language_features" / Path(cfg["reference_image"]).stem
)
prototype = torch.tensor(
    np.load(str(prefix) + "_f.npy")[selected["feature_index"]], device="cuda"
)
prototype = prototype / prototype.norm()
records = []
start = time.time()
for level in (1, 2, 3):
    model, iteration = torch.load(
        root / f"outputs/gsle/langsplat_{level}/chkpnt30000.pth", map_location="cpu"
    )
    assert iteration == 30000 and len(model) == 13
    for index in range(1, 7):
        assert torch.equal(source[index], model[index]), (
            "Source geometry changed",
            level,
            index,
        )
    field = model[7].detach()
    scores = []
    affinities = []
    floor_scores = []
    with torch.no_grad():
        for chunk in field.split(32768):
            latent = chunk.cuda()
            # Match gaussian_renderer: raw trainable language vectors are
            # normalized BEFORE splatting/decoding, not used at their raw norm.
            latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-9)
            features = decoder.decode(latent)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scores.append(
                torch.stack(
                    [clip.get_relevancy(features, i)[:, 0] for i in range(len(texts))],
                    1,
                )
                .cpu()
                .numpy()
            )
            floor_scores.append(
                clip.get_relevancy(features, len(texts))[:, 0].cpu().numpy()
            )
            affinities.append((features @ prototype).cpu().numpy())
    scores = np.concatenate(scores)
    affinities = np.concatenate(affinities)
    np.savez_compressed(
        dest / f"language_query_level{level}.npz",
        scores=scores,
        prototype_affinity=affinities,
        floor_scores=np.concatenate(floor_scores),
    )
    records.append(
        {
            "level": level,
            "iterations": iteration,
            "source_parameters_identical": True,
            "gaussians": len(field),
            "score_min": scores.min(0).tolist(),
            "score_max": scores.max(0).tolist(),
        }
    )
    print(records[-1], flush=True)
    del model, field
(dest / "query.json").write_text(
    json.dumps(
        {
            "texts": texts,
            "reference_proposal": selected,
            "levels": records,
            "elapsed": time.time() - start,
            "semantic_rule": "LangSplat pairwise text-vs-generic-negative softmax, temperature10, worst negative",
            "prototype_usage": "saved diagnostic similarity only; 3D instance association uses the automatic reference segment mask plus the trained text score, without a manual ROI",
        },
        indent=2,
    )
)
