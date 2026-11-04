GSLE industrial equipment replacement
====================================

Status is determined by actual records/checkpoints, not by the presence of code.
This folder is an independent implementation derived from the WACV2025 paper
and supplement. No official GSLE source release was found on the author page.
See PAPER_NOTES.txt for equations, schedules, ambiguities and assumptions.
Machine-readable module contracts and coordinates are in interfaces.json.
An explicit CFG-conditioning ablation is available through 09_cfg_ablation.sh.
It shares the completed coarse model and input artifacts, but saves its own
2000-step refinement, full renders and metrics under outputs/gsle/ablations/
text_only_cfg. Set GSLE_GPU to an available device before launching it. Its
only training configuration change is retaining mask/context/depth conditions
in the empty-text branch. See PAPER_NOTES.txt item13 and its design.json.
GSLE_OUTPUT, GSLE_RECORDS and GSLE_CONFIG support isolated refinement variants;
the default remains the recorded primary configuration. Use separate output
and record roots together. Never silently mix an ablation into a primary run.

Authorized target: the complete black assembly and white support base visible
at approximately75s in the source video. Replacement: a yellow six-axis robot
arm mounted on a metal base. No old manual housing ROI is used.

Execution location
  SSH ubuntuA6000
  /mnt/sda/ubuntu/lkx/papers/paper_eli
  source scripts/common.sh
The existing reverse HTTP proxy is localhost:17891 on the server. Downloading
uses the user's local proxy, immutable HF revisions and SHA256 verification.
Environments: envs/gsle_langsplat and envs/gsle. Existing environments are base
read-only dependencies through venv system-site-packages; separate CUDA raster
extensions are installed in each new environment.

Stage scripts (run in order, inspect each output before continuing)
  bash gsle/00_prepare.sh
  bash gsle/01_environments.sh
  bash gsle/02_download.sh
  bash gsle/04_language_optimized.sh
  bash gsle/05_localize.sh
  bash gsle/06_anchor.sh
  bash gsle/07_coarse.sh
  bash gsle/08_refine.sh
  GSLE_GPU=3 bash gsle/09_cfg_ablation.sh
  envs/gsle/bin/python gsle/compare_cfg_runs.py
Then visually review both variants and write a new final_visual_review.json;
do not reuse the historical quality verdict for a new run. After that:
  envs/download/bin/python gsle/select_delivery.py text_only
  envs/download/bin/python gsle/package_results.py
If optimization is already complete and only export needs recovery, run
  bash gsle/10_export.sh refine
Keep active shell launchers unchanged until they exit. The primary run's
post-training shell-read issue and export recovery are recorded separately in
records/gsle/export_recovery.json; no optimization updates are repeated.
Set GSLE_GPU before stage scripts to choose the device. The language script
uses GPUs0,2,3 for independent feature shards and language-field scales. Check
availability on the shared server and change its explicit mapping as needed.
It preserves all306 training views, all three language scales and30000 updates
  per scale.44 original held-out cameras remain excluded from training.

Validation commands (same ROOT/HF_HOME as common.sh)
  envs/gsle/bin/python gsle/validate_renderer.py
  envs/gsle/bin/python gsle/validate_diffusion.py zero
  envs/gsle/bin/python gsle/validate_diffusion.py di
  envs/gsle/bin/python gsle/validate_compactness.py
  envs/gsle/bin/python gsle/preview_checkpoint.py outputs/gsle/coarse/checkpoint_003500.pth
  envs/gsle_langsplat/bin/python gsle/validate_ae_graph.py
The AE graph check requires the reference frame's real extracted features.
Run it before train_autoencoder.py. Its output is numerical validation, not a
replacement scene. Batch size64,100 epochs,Adam LR0.0007,architecture and losses
remain those of LangSplat. CUDA graphs remove launch overhead; floating-point
differences from eager Adam are quantified in ae_graph_validation.json.

Data/coordinate interfaces
  data/gsle_langsplat: full calibrated images from the306-camera training split;
  feature extraction uses938x528, matching the original3DGS half resolution.
  language_features/*_s.npy: int32[4,H,W], -1 means no SAM segment.
  language_features/*_f.npy: float32[N,512], normalized OpenCLIP embeddings.
  language_features_dim3: identical maps plus scene-AE-compressed[N,3] features.
  outputs/gsle/langsplat_1..3/chkpnt30000.pth: original source parameters and
  learned3D language features; exact parameter/index identity is asserted.

  localization/query.json and language_query_level*.npz: decoded3D semantics.
  localization/target.json: COLMAP world coordinates, world-up estimate,
  bounding-box corners, reference camera and provenance of automatic selection.
  target_indices.npy/keep_indices.npy: original PLY row indices, no remapping.
  localization/target.ply and scene_without_target.ply: actual isolated/removed
  Gaussian regions. Reference proposals are diagnostic2D instance association;
  the trained3D language field is mandatory for final localization.
  localization/multiview/contact_sheet.jpg uses automatically selected visible
  original cameras. Floor fit diagnostics and the first bbox-only attempt are
  retained. Previously occluded floor is not invented by this implementation.
  localization/edit_region.json is a separate automatically derived upright
  generation box. Old-device deletion indices remain in target_indices.npy.
  AVP, object placement, DI-SDS masks and result orbits all use this same region.

  avp/candidates.json: all100 calibrated candidate cameras, visibility checks,
  eight-rotation brightness ratios, literal and text-consistent AVP rankings.
  avp/anchor.json, anchor.png, bbox_mask.png, depth.npz, depth_control.png:
  selected camera and inputs. Mask1/white means edit. Depth is volumetrically
  weighted Euclidean distance; control image is normalized and inverted1-d.

  inpainting/edited_anchor.png: actual SD1.5-inpainting/ControlNet output.
  inpainting/candidates and generation.json retain fixed-seed candidates and
  automatic semantic ranking. Failed setup attempts remain in outputs/gsle/attempts.
  06_generate.sh reruns generation/SAM using an existing calibrated AVP result.
  foreground/object_rgba.png, mask.png, reference_white.png: automatically
  extracted SAM foreground. Candidate masks and semantic ranking are retained.
  automatic_prompts.json records image-derived yellow foreground/background
  points used when box-only SAM also selected a cabinet. HSV supplies points
  only; the final mask is predicted by SAM. This prompt adapter is specific to
  the requested yellow object and can be replaced independently for other colors.

  coarse/final.pth: new-object parameters, optimizer state and invertible
  local-to-world similarity. final_object_local.ply is standalone local space;
  final_object_world.ply is already placed in the original scene coordinates.
  composition/scene.ply: coarse object plus unchanged retained background.

  refine/final.pth and final_object_*.ply: DI-SDS-refined object.
  final/scene.ply and object_world.ply: final scene and independent new object.
  composition.json verifies every retained background Gaussian bit for bit.

Camera convention: COLMAP world-to-camera R,t; camera+x right,+y down,+z forward.
All saved cameras have fx,fy,cx,cy,width,height. Object local coordinates use
x right,y up,z toward the anchor. Stable Zero123 uses DreamGaussian's opposite
elevation sign; the wrapper converts it explicitly. No implicit axis/scale swap.
The similarity scale is world units per local unit: the initial sphere diameter
(2*sphere_radius) maps to the largest generation-box side, not half that side.

Training and resuming
  gsle/config.json stores the effective paper settings and exposed assumptions.
  Coarse6500 iterations; checkpoint also saved at3500 because the supplement
  contains both counts. Refinement2000. Published batches6/4 are used when
  memory permits; --microbatch can accumulate the same effective batch with
  one optimizer update. Teachers/background remain frozen. Fixed-background
  shading is evaluated without autograd, while occlusion uses joint rendering.
  Checkpoints every500 updates include optimizer state. Resume with:
    envs/gsle/bin/python gsle/train_object.py coarse --resume PATH
  --steps is diagnostic-only and writes to coarse_diagnostic/refine_diagnostic;
  it never creates final.* in the actual experiment stage directory.
  Do not pass --steps to the composition wrapper scripts.

  Compactness densification ports the pinned GSGen formula including its
  published implementation's unusual surface-distance exponent. KNN is exact
  Euclidean cKDTree; the port is compared against extracted upstream functions.
  DI-SDS uses nine UNet input channels and four ControlNet latent channels.
  Equation7's unconditional expression omits all conditioning; the primary
  implementation zeroes mask/context/depth residuals and uses empty text.
  This channel-padding detail is not specified by the paper and is recorded.
  A text-only CFG alternative is exposed for explicit ablations, not silently
  substituted. View text uses the pinned SD1.5 text encoder.
  Frozen diffusion modules use FP16; Gaussian parameters/rendering and final
  SDS loss accumulation use FP32. SDS weighting is1-alpha_cumprod(t), as in
  the reused DreamGaussian guidance. These execution choices are disclosed
  because GSLE does not specify precision or its SDS weighting function.

Rendering
  render_results.py produces actual Gaussian renders of the original camera
  path plus a separate360-degree object/scene orbit. The source video lasts
  174.89655s.350 frames at2fps yield approximately175s of original-speed viewing;
  the12fps preview is deliberately6x faster (~29.17s), not the source duration.
  RGB comparisons outside the old/new-object projection union are recorded.
  Exact background parameter preservation does not imply unchanged occlusion.
  final/review/matched_views_1.jpg and matched_views_2.jpg compare original,
  coarse and refined scenes from eight identical cameras. summary.json records
  actual update counts, timings and background error on train/test cameras.
  These metrics are not a substitute for replacement ground truth, which does
  not exist for this scene; inspect the independent object and scene orbits.

Execution adaptations and records
  lang_features.py is a resumable streaming adapter around official multiscale
  SAM/OpenCLIP extraction. Binary-mask NMS is vectorized, SAM region cleanup
  runs in an ordered thread pool, and image crops are taken before masking.
  Actual-mask equivalence checks are saved; thresholds and generated pixels
  are unchanged. No frame count, scale count or training schedule is reduced.
  All current execution logs: logs/gsle. Text records: records/gsle.
  Upstream commits, model revisions/files and environment freezes are saved.
  Model weights, datasets, PLYs, videos and full logs are not committed to Git.
  After all stages complete, package_results.py creates gsle_results.tar.gz
  and its SHA256. The bundle contains final/intermediate Gaussian artifacts,
  calibrated cameras, images/videos, source, configs and logs. Large LangSplat
  feature/training files and pretrained weights remain in their server folders;
  package_manifest.json lists contents and exclusions explicitly.
  Extract this bundle into an existing paper_eli checkout. The base GaussianEditor
  and EditSplat environments are project dependencies of the two new venvs;
  their setup scripts remain in the repository. common.sh is also in the bundle.
  See THIRD_PARTY.txt and accompanying licenses for reused components.

Completed experiment (2026-11-05)
  See REPORT.txt for actual results and substantial visual limitations.
  Coarse6500 and both DI-SDS2000 variants completed; all decoded video, PLY,
  retained-background and image-count checks passed. The selected text_only
  variant is the better actual result, not a high-quality six-axis reconstruction.
  Main drop_all_eq7 produces severe yellow fog and is retained as a failure.
  Eight matched directions and twelve object views expose side/back artifacts.
  records/gsle/deliverables.json indexes reviewed files with SHA256 checksums.
  On Windows: powershell -ExecutionPolicy Bypass -File gsle/download_results.ps1
  Add -PreviewOnly to skip PLY/checkpoint files. Default destination: results/gsle.
  Full archive includes both variants, module artifacts, cameras, source and logs.
