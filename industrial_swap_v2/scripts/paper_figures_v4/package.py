"""Validate publication artifacts, make a static review page and two delivery archives."""
import csv,html,json,zipfile,shutil
import numpy as np
from figure_common import *
FIGURES={'F08':'F08_localization','F09':'F09_text_control','F11':'F11_m2_ablation','F12':'F12_m3_training','F13':'F13_spatial_alignment','F14':'F14_background_refinement','F15':'F15_efficiency','F16':'F16_failure_cases'}

def main():
    audit=[]
    for key,name in FIGURES.items():
        d=OUT/name
        for f in [f'figure_{key}.png',f'figure_{key}.pdf','data.csv','caption.txt','sources.json']:
            assert (d/f).is_file(),(key,f)
        im=Image.open(d/f'figure_{key}.png');dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-300)<1
        assert (d/f'figure_{key}.pdf').read_bytes().startswith(b'%PDF-')
        assert list((d/'panels').glob('*.png'))
        audit.append(dict(figure=key,png=source(d/f'figure_{key}.png'),pdf=source(d/f'figure_{key}.pdf'),pixels=im.size,dpi=dpi,rows=len(list(csv.DictReader((d/'data.csv').open())))))
    for name in ['T01_overall.csv','T02_ablation.csv','T03_efficiency.csv','tables.tex']:assert (OUT/'tables'/name).is_file()
    loc=read(OUT/'F08_localization/localization_metrics.json');m2=read(OUT/'F11_m2_ablation/metrics.json')
    reg=read(OUT/'F13_spatial_alignment/ablation_results.json');placement=read(OUT/'F09_text_control/placement_results.json')
    runtime=list(csv.DictReader((OUT/'tables/T03_efficiency.csv').open()))
    def mean_loc(m):return np.mean([r['reference_sam_iou'] for r in loc if r['method']==m])
    variant_lines=[]
    for v in placement['variants']:
        t=v['transform'];variant_lines.append(f"- {v['variant']}: {v['gaussians']:,} Gaussians, scale={t['scale']:.5f}, yaw={t['yaw_degrees']:.2f} deg, Anchor IoU={t.get('metrics',{}).get('mask_iou','NA')}, alignment gate={t.get('quality_gate_passed',True)}. Scene: `{v['scene_path']}`.")
    summary=f'''# Paper figures v4: actual experiments and limitations

## Delivery

Eight figures (F08, F09, F11-F16) contain PNG at 300 dpi, PDF, individual panels, data.csv, caption.txt and sources.json. T01-T03 are CSV plus tables.tex. F05/F06/F07/F10 in paper_figures_v3 were not changed.

Remote root: `{OUT}`

Local mirror: `E:\\codex_projects\\paper_eli\\industrial_swap_v2\\outputs\\paper_figures_v4`

`paper_figures_v4_publication.zip` contains figures, panels, data, English captions, manifests, tables and reproduction scripts. `F09_generated_assets.zip` contains the actual new GLBs, local/world Gaussian assets, four composed scene PLYs, images, camera/placement records and source pointers. Full optimization checkpoints/logs remain in this experiment directory on the server.

## New executions

- F09: three original task-file descriptions (SCARA, CNC machining center, single-deck roller conveyor); seven successful Qwen generations in total including four F11 controls. Three initial F09 transport failures occurred while the SSH proxy forward was unavailable; preserved attempt records document restoration and explicit recovery. No output was substituted from image synthesis outside Qwen or manually painted.
- F09: three official TRELLIS.2 1024-cascade runs, 36 PBR views per asset, 4,000 RGB+Depth Gaussian-training steps each, yaw/scale placement with fixed Case00 installation center, support height and allowed volume. All four scene compositions use the identical original retained M1 background. No background optimization was run for these variants.
- CNC official GLB export at one million target faces stalled in UV chart computation. Its original native mesh and log were retained; only the official export decimation target was changed to 250,000 faces. No diffusion sampling was repeated. Exact adaptation is recorded in `F09_text_control/variants/C_cnc/records/export_resource_adaptation.json`.
- F13: new actual-Gaussian bbox and silhouette-only controls for Case00/02; the full method uses its existing recorded transform. A real sampled yaw-scale objective surface was rendered.
- F14: one new Case00 RGB-only background control, 2,000 steps, same frozen device/pose, camera masks, seed and learning rates as the original full M4 run. Visibility/occlusion-specific losses disabled. No M1-M3 regeneration for this control.
- F08/F12/F15: existing records and assets reused; no LangSplat retraining. T01 uses new read-only renders of existing final scenes.

## Findings that must remain visible in the paper

1. F08 mean input-SAM agreement IoU: text-only {mean_loc('gsle'):.4f}, photo-assisted {mean_loc('ours'):.4f}. Case10 contributes the largest instance-selection difference; Case02 is slightly worse with photo assistance. SAM participated in the assisted localization, so these values are **not independent ground-truth accuracy**. Held-out manual annotations and correct-instance success labels were unavailable.
2. F09 changes device appearance/category at the fixed workstation, but the SCARA request yields an articulated-arm appearance whose kinematic topology is not verified. CNC foreground extraction removes part of the side control display; the actual imperfect input/output is retained. Shared-background recomposition of A deliberately differs from the optimized scene in F05/F10, while preserving the original robot geometry and pose.
3. F11 all four single-seed outputs are similar. Inside-edit fractions: {', '.join(f"{r['condition']}={r['object_inside_edit_fraction']:.4f}" for r in m2)}. Outside-edit RGB MAE: {', '.join(f"{r['condition']}={r['background_rgb_mae']:.5f}" for r in m2)}. C4 has the lowest measured background change in this one run; C3 has slightly higher inside-mask containment. Neither establishes statistical superiority. Mechanical axis count is not verifiable from a single image.
4. F12 shows improved fidelity to the existing GLB/PBR RGB-D targets on six held-out views per asset. This does not measure real geometric truth or relighting accuracy. No RGB-only M3 experiment is claimed.
5. F13 silhouette and full-registration results are close for these two assets. Both outperform naive zero-yaw bbox placement in mask overlap. Center offset measures deviation from the installation prior, not pose ground-truth error; sampled Gaussian covariance support outside the box is not volumetric intersection.
6. F14 full M4 improves trusted device visibility much more than RGB-only, while RGB-only better preserves the measured environment RGB. Both 2,000-step candidates fail an existing per-view preservation guard. Their guarded exports retain Step 0. Candidate and guarded final are explicitly separated.
7. F15/T03 recorded component subtotals are {min(float(r['Measured_subtotal_s']) for r in runtime)/60:.2f}-{max(float(r['Measured_subtotal_s']) for r in runtime)/60:.2f} minutes. These are lower bounds, not complete end-to-end times. Manual Qwen latency, PBR view/export time and some preparation overhead lack consistent instrumentation. Missing values are NA. One-time source 3DGS training is reported separately; no CPU RSS is mislabeled as GPU memory.
8. F16 documents actual M1 over-selection relative to input SAM, missing CNC control-display foreground, and retained old-footprint background defects. No failed scene was fabricated.

## T01 interpretation

GSLE uses the coarse scene, matching the current F07 display. Ours uses the existing 2,000-step candidate scenes, with Case07's accepted yaw adjustment; these candidates failed their automatic preservation guard. Background LPIPS uses the same three camera positions, resolution and common old/new Ours exclusion masks for every method within a case. This is separate from viewpoint-matched qualitative figures. A method editing another area can score near zero in these cameras without succeeding at replacement (notably GSLE Case10). Report localization and visibility context together with preservation scores.

Independent Localization IoU, Replacement Success Rate and Text Attribute Accuracy remain NA. Reference-SAM overlap is a separate diagnostic. Workspace overflow is explicitly an opacity-weighted fraction of object Gaussian **centers** outside the intended common workstation, not exact occupied volume; DGE has no independent replacement object and is NA.

## F09 asset results

{chr(10).join(variant_lines)}

## Reproduction

Scripts: `industrial_swap_v2/scripts/paper_figures_v4/`.
Run from the remote project root, using `envs/gsle/bin/python` except SAM evaluation with `envs/gsle_langsplat/bin/python`:

```bash
python industrial_swap_v2/scripts/paper_figures_v4/figure_f12.py
python industrial_swap_v2/scripts/paper_figures_v4/run_alignment_ablation.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f13.py
python industrial_swap_v2/scripts/paper_figures_v4/run_rgb_only.py
python industrial_swap_v2/scripts/paper_figures_v4/render_f14.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f14.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f15.py
python industrial_swap_v2/scripts/paper_figures_v4/run_f09_generation.py
python industrial_swap_v2/scripts/paper_figures_v4/place_f09.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f09.py
python industrial_swap_v2/scripts/paper_figures_v4/evaluate_localization.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f08.py
python industrial_swap_v2/scripts/paper_figures_v4/run_f11_generation.py
python industrial_swap_v2/scripts/paper_figures_v4/evaluate_f11.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f11.py
python industrial_swap_v2/scripts/paper_figures_v4/figure_f16.py
python industrial_swap_v2/scripts/paper_figures_v4/evaluate_overall.py
python industrial_swap_v2/scripts/paper_figures_v4/make_tables.py
python industrial_swap_v2/scripts/paper_figures_v4/package.py
```

Generation runners use durable request fingerprints and reuse saved responses. Do not delete submission locks. The RGB-only launcher requires a fresh output directory; it is not a command to overwrite the completed experiment. GPU selection/resource availability must be checked when reproducing. All numerical and image-source records are preserved in figure sources.json and experiment subdirectories.
'''
    (OUT/'experiment_summary.md').write_text(summary,encoding='utf-8')
    cards=[]
    for key,name in FIGURES.items():
        caption=(OUT/name/'caption.txt').read_text()
        cards.append(f'<section id="{key}"><h2>{key}</h2><a href="{name}/figure_{key}.png"><img src="{name}/preview_{key}.jpg" loading="lazy"></a><p><a href="{name}/figure_{key}.pdf">PDF</a> · <a href="{name}/figure_{key}.png">300 DPI PNG</a> · <a href="{name}/data.csv">Data CSV</a> · <a href="{name}/caption.txt">Caption</a> · <a href="{name}/sources.json">Sources</a></p><p>{html.escape(caption)}</p></section>')
    page='''<!doctype html><meta charset="utf-8"><title>Paper Figures v4</title><style>body{font-family:Arial,sans-serif;max-width:1400px;margin:24px auto;padding:0 24px;background:#f5f6f8;color:#20252b}nav{position:sticky;top:0;padding:15px;background:#fff;border-bottom:1px solid #ddd}nav a{margin-right:22px}section{background:#fff;margin:24px 0;padding:24px;border-radius:8px}img{width:100%;max-height:1100px;object-fit:contain}a{color:#1b5894}p{line-height:1.55}.note{background:#fff3d2;padding:16px}</style><h1>SCI 实验图检查 · Paper Figures v4</h1><p>真实实验结果、高清图、数据与来源。F05/F06/F07/F10 保持原样。</p><p class="note">F08 使用输入 SAM 一致性诊断；F09 保留生成与分割缺陷；F14 候选未通过 Guard，Guarded Final 回退 Step 0；缺失评价与完整耗时均标为 NA。</p><nav>'''+''.join(f'<a href="#{key}">{key}</a>' for key in FIGURES)+'''</nav><p><a href="paper_figures_v4_publication.zip">论文图表压缩包</a> · <a href="F09_generated_assets.zip">F09 三维结果压缩包</a> · <a href="experiment_summary.md">实验总结</a> · <a href="tables/tables.tex">LaTeX 表格</a></p>'''+''.join(cards)
    (OUT/'index.html').write_text(page,encoding='utf-8');save(OUT/'delivery_validation.json',dict(figures=audit,figure_count=8,completed_figures=list(FIGURES),external_old_figures_modified=False))
    scripts=BASE/'scripts/paper_figures_v4';dest=OUT/'reproduction_scripts';shutil.copytree(scripts,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    with zipfile.ZipFile(OUT/'paper_figures_v4_publication.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in FIGURES.values():
            d=OUT/name
            for p in d.iterdir():
                if p.is_file() and (p.name.startswith(('figure_','preview_')) or p.name in ['data.csv','caption.txt','sources.json']):z.write(p,p.relative_to(OUT))
            for p in (d/'panels').glob('*.png'):z.write(p,p.relative_to(OUT))
        for directory in [OUT/'tables',dest]:
            for p in directory.iterdir():
                if p.is_file():z.write(p,p.relative_to(OUT))
        for name in ['index.html','experiment_summary.md','delivery_validation.json']:z.write(OUT/name,name)
    # All new final assets, not enormous intermediate checkpoints or training-view caches.
    selected=[];f09=OUT/'F09_text_control'
    for v in placement['variants']:
        p=f09/'variants'/v['variant']
        selected.extend(x for x in p.iterdir() if x.is_file() and x.suffix in ('.ply','.png','.json','.yaml'))
        for sub,names in {'m2':['edited_scene.png','object_rgba.png','object_mask.png','generation_prompt.txt','anchor_camera.json'],
                'm3':['trellis_asset.glb','object_refined_local.ply','pbr_training_summary.json','trellis_generation.json'],
                'm4':['object_world.ply','transform.json']}.items():
            selected.extend(p/sub/n for n in names if (p/sub/n).is_file())
    selected += [f09/'fixed_installation_space.json',f09/'placement_results.json']
    with zipfile.ZipFile(OUT/'F09_generated_assets.zip','w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in selected:z.write(p,p.relative_to(f09))
        original_object=Path(placement['variants'][0]['object_path']);z.write(original_object,'variants/A_robot/object_world.ply')
    print('DELIVERY COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
