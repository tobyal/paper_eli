# GaussianEditor native complete-flow baseline

This entry is independent of the earlier spatially associated baseline. It accepts only the original Gaussian scene, full original COLMAP dataset and the requested A/B texts. No `target_geometry.json`, Ours semantic mask, Ours camera shortlist or previous generated asset enters the workflow.

1. `prepare.py case00` creates an isolated source copy and an allowlisted case configuration.
2. `run.py case00 --gpu 2 --through localize` calls the upstream text segmentor and semantic-tracing method. Native defaults are 24 uniformly sampled cameras (seed 0), first returned LangSAM candidate, and threshold 0.7. All other candidates are retained for inspection.
3. The Add camera is selected from the full registered camera set using only the visible contribution of GE-selected Gaussians. A thresholded visible mask's bounding rectangle is expanded by 8% on each side. This scripts the original interactive camera/rectangle controls; it is not a new localization network.
4. Continuing `run.py case00 --gpu 2` executes the actual native Delete, Add, Wonder3D, NeuS, mesh-to-Gaussian/IP2P and depth-placement code. Delete reproduces the UI callback defaults: 24 cameras, 1500 steps, densification interval 50. No spatially selected camera shortlist is used.
5. Add uses native DPT placement with the default Depth Scale of 1.0. Native interactive scale adjustment can be explicitly recorded in a separate experiment, but no Ours depth or registration is available here.
6. `report.py` exports images, a summary and an offline inspection package. PLYs and heavy intermediates remain on the server.

The upstream Add refines the standalone object **before** insertion. There is no additional post-insertion refinement stage. The requested compatibility filename `scene_inserted_before_refine.png` is explicitly documented as the first inserted scene, not evidence of a nonexistent subsequent training stage.

The native model/runtime compatibility patches from the previous isolated environment are retained. `threestudio/utils/sam.py` is restored byte-for-byte from the pinned official commit; the externally supplied instance selector is removed. Semantic weight dumps and intermediate images are export-only instrumentation. Original experiments remain unchanged.

Case00 was completed first. The user subsequently authorized the same native flow for case02, case04, case07 and case10. `run_remaining.py` schedules those four cases on two available GPUs without changing their algorithm/configuration. `report.py` produces the combined five-case offline inspection package; the original case00 package is preserved.

## Audit of the previous baseline entry

| Input/control | Previous baseline adapter | This native-flow entry |
|---|---|---|
| Localization cameras | 20 views ranked using Ours old-device masks | Full original COLMAP set; official random 24-view tracing |
| LangSAM instance | Candidate ranked against external `target_geometry` projection | Exact official segmentor, first returned candidate; all candidates exported |
| Tracing threshold | 0.5 | Official initial value 0.7 |
| Delete controls | 20 views, densification interval 100 | Native Delete callback: 24 views, interval 50 |
| Add rectangle | Projection of externally supplied 3D target box | GE's own rendered visible mask rectangle plus 8% margin |
| Placement depth | External target depth controls the slider | Native DPT estimate, GUI default slider 1.0 |
| Generated assets | Earlier same-baseline recovery could reuse cached assets | Fresh image, Wonder3D, NeuS and mesh-to-Gaussian generation |

The native camera sampling, first-mask choice and DPT estimate can fail in this dense industrial scene. Their actual failures remain in the stage images; this adapter does not silently substitute a better mask, foreground, pose or generated device.
