# Weak DI-SDS (independent conservative experiment)

Latest requested variant: rerun **case00 only**, from its original Coarse, with
`--sds-weight 0.02 --iterations 1000`. The first 0.0001 case00 output is retained
under `gsle_weak_disds/superseded/`; case00 now contains the 0.02 run. The previous
five-case queue was stopped at the user's request. Geometry/opacity remain
frozen and color LR remains 0.001. `pack_case.py case00` creates a single-case
review package. Details below describe the initial 0.0001 experiment; the
actual config and verification JSON always record the run's selected weight.

This experiment does not replace the paper-configuration GSLE baseline. Inputs
come directly from `baseline_comparison/gsle_native/<case>/coarse/final.pth`.
Localization, AVP, inpainting, SAM foreground, Coarse, and original refinement
remain unchanged. Each output lives under `baseline_comparison/gsle_weak_disds/`.

Only `color_logit` is trainable. The optimizer has one Adam group, learning rate
0.001. Positions, log scales, rotations, opacity logits, Gaussian count, and
world transform are frozen. SDS weight is 0.0001 (native: 0.1). Native diffusion
weights, text guidance, timestep sampling, depth conditioning, inpainting mask,
DI-SDS equations, random camera sampler, effective batch of four, and Anchor
RGB/Mask weights of 1500 remain unchanged. The Mask loss is still evaluated;
its gradient cannot modify frozen shape or opacity.

`train.py` is a minimal copy of the native trainer with color-only permissions
and safe frozen-gradient statistics. Each run snapshots native dependencies,
records the exact training diff, source SHA256 values, configs and command lines,
then asserts geometry and retained background invariance during export.

Run on the server (select a free GPU):

```bash
envs/gsle/bin/python baseline_comparison/weak_disds/test_policy.py
envs/gsle/bin/python baseline_comparison/weak_disds/run.py case04 \
  --gpu 3 --iterations 500 --label pilot_case04_500
```

Inspect the pilot `progress.png`, `comparison.png`, `scene_comparison.png`,
`multiview.png`, and `records/verification.json`. Only after inspecting the
500-step output, save a truthful `records/gate.json` with `passed: true` and the
visual observations. This gate is an experiment quality check, not a requirement
for an additional user approval.

```bash
envs/gsle/bin/python baseline_comparison/weak_disds/run.py case04 --gpu 3 \
  --resume-pilot baseline_comparison/gsle_weak_disds/pilot_case04_500
envs/gsle/bin/python baseline_comparison/weak_disds/run.py case00 --gpu 0
# Run case02, case07, case10 similarly, respecting shared GPU availability.
```

case04 continues the accepted pilot from step 500 to a total of 1000, preserving
Adam and RNG state. Other cases start from their original Coarse checkpoint.
The pilot is preserved in its own directory. Existing experiment directories
are never overwritten by the launcher.

Outputs per case:

- `refine/previews/reference_000100.png`, `reference_000500.png`,
  `reference_001000.png`: actual post-update Anchor renders.
- `refine/final.pth`, `final_object_local.ply`, `final_object_world.ply`.
- `final/scene.ply`: retained native background plus the weak-refined object.
- `comparison.png`, `scene_comparison.png`: Coarse / original DI-SDS 2000 /
  weak DI-SDS 1000, using the same Anchor camera.
- `multiview.png`, `scene_multiview.png`: original calibrated review cameras.
- `records/verification.json`: parameter equality, unchanged Alpha/depth,
  background equality, color changes, reference MSE, and exact output paths.
- `records/source_preservation.json`: original input and baseline hashes checked
  again after training and export.

Fixed geometry guarantees silhouette preservation but does not repair malformed
Coarse geometry or missing components. Lower RGB MSE alone does not establish
perceptual improvement; inspect the actual images and report limitations.
