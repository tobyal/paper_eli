# DGE native complete-flow baseline

`prepare.py` starts from official git commit `7eed0bc57824707c53b935bd19ea994f19f41eae`, restoring the exact official LangSAM segmentor. Runtime fixes retained: dataclass default factory, arccos numerical clipping, local rasterizer index mapping, and CLIP maximum-length truncation. The patch is saved per experiment.

Inputs are only the original 3DGS, full 350-view COLMAP dataset and the exact A/B texts. `target_record`, instance geometry, SAM instance, Ours camera lists and assets are never read. Official `configs/dge.yaml` decides sampling, threshold, losses, densification and schedule: 20 random views, seed 0, threshold 0.8, 1000 fitting steps, view refresh/edit at steps 0 and 500. `local_edit` stays at its official false default; local updates are enforced by the official Gaussian gradient mask.

Run on server: `python3 baseline_comparison/native_dge/run.py case00 --gpu 2`. The native audit exports candidates without reranking and records actual traced indices, edited views, live step-500 render and final PLY. Visualization-camera selection happens after localization and affects only exports. DGE has no Delete/Add or separate object insertion stage.

Outputs are under `baseline_comparison/dge_native/caseXX`. A completed process is not a claim of successful geometric replacement. Existing controlled-localization experiments remain unchanged.
