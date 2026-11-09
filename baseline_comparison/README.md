# Industrial replacement baseline experiments

All experiments execute under `/mnt/sda/ubuntu/lkx/papers/paper_eli` on SSH host `ubuntuA6000`. The five physical instances and exact current A/B instructions are in `prompts.json`. Existing Ours results and original `gsle/` are read-only inputs. No Qwen or TRELLIS is called by this baseline workflow.

## Methods

| Method | Native pipeline | Training |
|---|---|---|
| GaussianEditor | LangSAM semantic tracing → Delete with ControlNet inpainting → rectangular Add → rembg → Wonder3D six RGB/normal views → NeuS mesh → mesh-to-Gaussian/IP2P → native depth-slider placement | Delete 1500; NeuS 3000; mesh-to-Gaussian 1800 |
| DGE | Documented local editing: `seg_prompt=A`, full-scene rendering, semantic Gaussian gradient masks, multiview IP2P guidance | 1000 |
| GSLE | Existing paper-derived LangSplat → AVP → depth-conditioned inpainting → generic SAM foreground → Stable Zero123 3D-SDS → DI-SDS → composition | Coarse 6500; refinement 2000 |

GSLE is **not** an official public implementation. Its diffusion/optimization implementation is copied from the existing project; `vendor/gsle/train_object.py` and `diffusion.py` retain the source algorithms. Per-case reference SAM regions only associate the requested physical instance; fresh LangSplat queries derive each 3D localization.

## Execution

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
python3 baseline_comparison/scripts/run_ge.py case00 --gpu 1
python3 baseline_comparison/scripts/run_dge.py case00 --gpu 0
python3 baseline_comparison/scripts/run_gsle.py case00 --gpu 2
CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python baseline_comparison/scripts/render_comparison.py
python3 baseline_comparison/scripts/build_report.py
```

Select an available GPU after inspecting shared occupancy. Do not stop other users' work. The launchers retain stage logs/status and never label queued or failed work as complete. Archive a failed DGE attempt before retrying. Partial GSLE runs use `--through foreground` or `--through coarse`; omit this option to continue completed stages into the full pipeline.

## Input and compatibility records

- Source scene: original iteration-30000 reconstruction, with SHA256 in `prompts.json`.
- GaussianEditor and DGE use 20 target-observing original training cameras, selected by old-target visibility. Intrinsics/extrinsics and original training images are unchanged. Native LangSAM masks are recomputed; Ours segmentation masks do not replace them. Instance geometry ranks native detections to associate the intended instance.
- The native semantic trace threshold is 0.5 for this industrial scene. All prompt strings and the selected views are saved per case.
- GaussianEditor uses an isolated environment overlay with PyTorch Lightning 1.9.5, compatible with its bundled Wonder3D NeuS code. Model repository moves, exact revisions, and Omnidata checkpoint checksums are recorded under `records/`.
- The GE headless adapter invokes the real `WebUI.delete` and `WebUI.add` functions. Only display handles and interactive controls are automated. The Add rectangle comes from the old target's projected spatial region. Its native depth slider is set to the old target's camera depth; no Ours alignment, generated image, or new-device geometry is used.
- GE's unused SD2.1 prompt allocation and overwritten DPT ImageNet initialization are removed. Full native checkpoints still load strictly. The bundled NeuS launcher respects the parent GPU allocation. All source compatibility changes have patch records.
- Add camera selection requires the entire projected editing box to remain inside the original camera with a 4% margin. The first case00 boundary-clipped attempt is preserved under `attempts/` and excluded from the comparison. Mesh conversion uses a recorded seed of 42.
- The native Add placement equation and DPT alignment are retained. A positive depth slider value may exceed the GUI's original 5.0 upper bound when required by this scene's units (case00 needs about 5.89). This explicit interface-range adaptation is recorded in `add/placement.json`; it is not Ours pose optimization. Recovery reuses only the same baseline attempt's completed native image, mesh and Gaussian, after exact Add-input matching.
- NerfAcc 0.3.3 is compiled with CUDA 11.8 into the project's isolated Torch extension cache. The original shared environments are not rebuilt.
- DGE's documented local editing does **not** require the `local_edit=True` target-only rendering flag. Early diagnostic attempts with that flag are retained under `attempts/` and excluded from the final comparison. The local-raster index compatibility patch is retained, but the documented final workflow renders the complete scene and does not exercise that branch.
- GSLE generic foreground selection uses SAM box candidates, exact replacement-text CLIP relevance, SAM confidence and containment, without yellow/robot color rules. Inpainting and SDS are unchanged.
- Case02 GSLE's original AVP orbit had a maximum target visibility around 0.813, below the existing industrial safety gate of 0.9. An adjusted orbit was also tested and retained. The final case02 configuration restores the original orbit and uses a documented 0.8 gate, retaining 100 azimuths, eight image rotations and the original brightness scoring.
- Native CLIP/diffusion tokenizer length limits remain in effect. Full requested text is saved even where the native model truncates its token conditioning; warnings remain in logs.
- GSLE retains effective batches of six coarse views and four refinement views. Physical microbatches are two for coarse generation; refinement cases 00/04/07 accumulate one view at a time, while pending cases 02/10 use the native four-view batch after a memory-availability gate. This only schedules memory usage; iteration counts and loss weights are unchanged.
- DGE's prompt-embedding helper omitted `truncation=True`, which crashes on the long case04/case07 prompts before training. The compatibility patch applies the frozen encoder's existing token limit, logs full and retained token lengths, and preserves the full requested A/B strings.
- Historical Ours case02 B contains an additional placement-volume phrase. This is explicitly recorded as a text mismatch; its existing output is not falsely claimed to have been rerun with byte-identical text. Case07 uses the user-approved yaw correction. All Ours columns use actual 2000-step outputs; guard outcomes are retained in provenance.

## Results and interpretation

`comparison/index.html` links the unified text manifest, actual run records, PLY files, and three original-camera comparisons per case. Full-frame and identical target crops are both saved. Every rendered PLY receives a SHA256 and Gaussian count.

`training_complete` means a process completed and produced a real final PLY. It does not assert successful equipment replacement. Qualitative failures, poor geometry and foreground contamination must remain visible. Shared-GPU stage durations are operational timings, not a controlled efficiency benchmark.

The final common auxiliary cameras are selected only from original target visibility, angular diversity and full containment of the 1.25x old-target box with a 3% frame margin. Preliminary boundary-clipped camera comparisons are archived separately; no method-specific camera is used for the final figures.

`scripts/snapshot_reproduction.py` captures the exact executed source, complete cumulative patches against pinned upstream commits, per-case settings and run records. Use this snapshot for reproduction: the initial preparation scripts alone predate later compatibility fixes. Model weights and datasets remain at the paths recorded in the manifest.

`scripts/export_offline.py` packages the inspection page, full-resolution figures, prompts, logs and source snapshot as `comparison/offline_review.zip`. Extract it and open `comparison/index.html`; no SSH web tunnel is required. Large final scene PLYs stay on the server and their exact paths are shown in the offline page and experiment CSV. They can be downloaded with `scp -o ClearAllForwardings=yes ubuntuA6000:<absolute-server-path> <local-destination>`.

## Source control scope

The Git tree includes the five-case prompt manifest, launch/export/validation scripts, the exact case-parameterized GSLE source in `vendor/gsle`, pinned upstream commits, cumulative GaussianEditor/DGE patches, environment freezes and model revisions. Large third-party checkouts, weights, images, PLYs, caches and run logs remain on the server.

`records/GaussianEditor_final_cumulative.patch` and `records/DGE_final_cumulative.patch` are applied to clean copies of their respective recorded commits. They include all earlier compatibility patches; do not apply the partial patches on top of them. Keep upstream submodules and the documented project model-cache links available. `prepare_ge_copy.py` and the individual patch scripts document the original setup sequence; the cumulative patches are authoritative for the executed version.

The adapted GSLE copy is already tracked: do not rerun `prepare_gsle_copy.py` over it. After preparing case directories, restore each saved `configs/gsle/caseXX.json` to `gsle/caseXX/config.json` before launching to retain the effective AVP and generation settings. All data paths refer to the recorded server project layout. Full execution logs and output provenance are delivered separately in the offline inspection package.
