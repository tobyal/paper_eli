# Viewpoint-Matched Rendering

This read-only exporter renders the final Native GaussianEditor, DGE and GSLE
PLYs and the exact Ours paths listed in `baseline_comparison/prompts.json`.
It never runs a training, diffusion, segmentation or generation model and never
changes or rewrites a Gaussian PLY. GSLE means the original native DI-SDS output,
not the later color-only experiment. Source-file SHA256 values are checked before
and after rendering.

Run from the server project root, choosing an available shared GPU:

```bash
CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python baseline_comparison/matched_rendering/render.py --case case00
# Inspect case00 before running case02, case04, case07, case10 similarly.
envs/gsle/bin/python baseline_comparison/matched_rendering/report.py
```

Outputs go to `baseline_comparison/matched_views/`. Final individual images use
1024 x 1024 pixels, a single Gaussian rasterizer, the full stored SH appearance,
and identical rendering settings. Figure layouts use 300 dpi PNGs. Exact camera
intrinsics/extrinsics, source hashes, target evidence, selected candidates,
visibility and framing diagnostics, and crop boxes accompany every case.

Target definition:

- Ours: its own M1 center and exact saved M2 Anchor camera. The manifest's
  `ours_scene` is used as-is, including the recorded guard flag.
- GaussianEditor: native Add world-space object center, checked against
  `add/placement.json`. Failed placement is not corrected.
- DGE: native traced indices in the original scene. Where tracing selects
  disconnected objects, focus on the component with greatest actual Gaussian
  contribution in DGE's own original primary review camera. This selects a camera
  framing region only, not a new segmentation. All native selection statistics
  and the complete unchanged final scene are preserved.
- GSLE: keep its own LangSplat target-box center fixed. Use the actual final
  generated object to assess framing extent without recentering on outliers.

Baseline cameras transport the Ours camera-to-target rig to their own centers,
preserving view direction, roll, FOV, principal point, and normalized target
position. This is particularly important for Ours' off-axis case02 camera. A
camera is not incorrectly converted to a centered optical model. Candidates
adjust distance first, then at most 15 degrees of yaw and 6 degrees of pitch
when visibility/framing requires it. Original and Ours use the same exact main
camera. The two auxiliary Ours rigs use +/-20 degrees yaw; baselines match these.

Matching uses rendered target Alpha, an expanded canvas to measure edge clipping,
and actual foreground contribution in the full scene. DGE visibility uses its
original traced component as an explicitly recorded proxy, because final Gaussian
indices may change during native optimization. No final DGE point identity is
assumed. RGB image quality or semantic success is not a camera-selection score.

The panels are **Viewpoint-Matched Rendering**, never a claim of pixel-aligned or
same-camera comparison across independently localized targets. Legitimate
occlusion, source reconstruction defects, generation failures and scene artifacts
remain visible. The per-baseline original/final controls use identical cameras.

## Red localization figures

`localization_red.py` renders the PRE-EDIT source scene, overriding the rendered
colors of each method's actual saved localization indices with pure red. It uses
all indices, including disconnected false positives: GE's
`localization/selected_indices.npy`, DGE/GSLE's `target_indices.npy`, and Ours'
M1 `target_indices.npy`. It does not use generated-object silhouettes or fill
projected boxes. Geometry, opacity, depth sorting and nonselected colors are
unchanged; no PLY is written. Source hashes and zero image differences outside
selected-Gaussian visible contribution are checked.

```bash
CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python baseline_comparison/matched_rendering/localization_red.py
CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python baseline_comparison/matched_rendering/localization_native_views.py
```

Outputs are separate in `baseline_comparison/matched_views_localization_red/`:
previously matched cameras, a common Ours-anchor diagnostic, two auxiliary views,
and each method's saved native localization reference camera. Native reference
views preserve their original aspect ratios and are letterboxed in figure panels.
The native-camera supplement is useful when a method's generated object's
placement differs from its original localized target. Its views are explicitly
labeled as different, not viewpoint matched. The red areas represent localization
of the old equipment, not the newly generated assets or the final editing extent.

## Per-method target-centered localization/final pairs

`target_centered_pairs.py` implements the later display policy: prioritize each
method's own generated target and native generation/reference direction, rather
than matching the Ours direction. GE starts from its native Add camera, GSLE from
its AVP generation camera, and DGE from its own primary localization camera. DGE's
native-visible framing component is reused from the saved camera audit, while all
native target indices remain in the red localization images. Ours keeps its saved
final anchor. A baseline can reuse that exact Ours camera only when the native
direction differs by at most 18 degrees, the target bearing is suitable, and
rendered visibility/framing are comparable to its independently selected camera.

Each method's pre-edit original, red localization and actual final PLY use
**identical intrinsics and extrinsics**. Auxiliary views orbit each method's own
chosen rig. Per-method crops are identical between localization and final. This
is not a cross-method viewpoint-matching claim; native directions may differ.
No localization indices, Gaussian parameters or source files are changed.

```bash
CUDA_VISIBLE_DEVICES=2 envs/gsle/bin/python baseline_comparison/matched_rendering/target_centered_pairs.py --case case00
# Run case02, case04, case07, case10 similarly, then assemble:
envs/gsle/bin/python baseline_comparison/matched_rendering/target_centered_pairs.py --package
```

Outputs are separate in `baseline_comparison/target_centered_views/`, with paired
figures, individual stages, two auxiliary pairs, exact cameras, candidate scores,
source hashes and a self-contained review archive. Source PLY SHA256 values are
verified before and after; zero training and zero PLY writes are required.

### Case00 common-camera revision

`case00_ours_view.py` applies the user's case00-only display override: every method
uses the previous Ours main and auxiliary cameras exactly. Red localization is
rendered with `0.6 * red + 0.4 * native_color` on selected Gaussians; physical
opacity remains unchanged. Shared crop coordinates are copied from Ours. This
reveals actual placement differences rather than moving any generated asset.
Outputs are in `target_centered_views/case00_ours_view_v2/`. The gallery points
case00 to this revision; previous case00 files and all other cases remain intact.

`case02_ours_view.py` applies the same requested display revision only to case02,
including its off-center camera intrinsics and both saved Ours auxiliary views.
It writes `case02_ours_view_v2/` and updates the case02 overview row starting from
the accepted case00 overview, preserving case00 and all other case rows.

`case04_view_adjust.py` applies the selective case04 revision: GSLE shares the
exact saved Ours cameras; DGE and Ours keep their cameras and final image pixels.
`case04_camera_search.py` frames GE using its native-visible localization component
and generated asset, searching around the Ours viewing direction. All native GE
indices remain in the red visualization. Real localization/final visibility and
image-edge margins guide camera choice; source models are never transformed.
`--refresh-camera` reevaluates cached candidates and refreshes only this display
revision, without repeating localization, training, or candidate rendering.
Results are in `case04_view_adjust_v2/`; existing case00/case02 revisions and all
other overview rows remain unchanged.

The later `case04_ge_object_view.py` correction centers GE on its actual generated
`object_world.ply`, retaining the previous viewing direction where possible.
Old localization visibility no longer affects its camera score. It writes
`case04_ge_object_view_v3/`, copies DGE/GSLE/Ours images byte-for-byte from v2,
and renders only GE; its red localization and final result still share cameras.

`case04_ge_joint_view.py` implements the subsequent joint-framing correction in
`case04_ge_joint_view_v4/`. `case04_joint_framing.py` requires both the original
localized region and generated object to be inside the camera with image margins,
using actual Gaussian alpha and scene contribution to evaluate candidates.
Main and auxiliary localization/final pairs share cameras. DGE, GSLE and Ours
images are copied unchanged from v3; no source Gaussian parameters are modified.

`gsle_view_adjust.py --case case07` reuses all three Ours cameras for GSLE.
`--case case10` preserves Ours orientation and intrinsics, transports the camera
to GSLE's native target and chooses a wider distance from actual alpha/visibility.
GE, DGE and Ours images are copied byte-for-byte. `--publish` updates only GSLE's
case07/case10 overview columns and links; case04 already shares GSLE/Ours cameras
in its accepted v4 revision. All source models and earlier display revisions remain.

`comparison_gsle_coarse_v8.py` replaces only the GSLE display column in the
accepted five-case v7 package. It renders each saved `composition/scene.ply`
from GSLE Coarse 6500 with the exact saved main/auxiliary cameras, verifies its
tail against `coarse/final_object_world.ply`, and labels every figure GSLE Coarse.
Localization and other methods' images remain byte-identical. `--publish`
updates the current `comparison_updated_v6/index.html` entry to the v8 package,
preserving its previous HTML and all previous images. No training or PLY writes.
