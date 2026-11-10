"""Isolated paper-derived GSLE, retaining the existing generation kernels."""
import argparse,difflib,hashlib,json,shutil
from pathlib import Path

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
CODE=BASE/'vendor/gsle_native';SOURCE=BASE/'vendor/gsle'
REUSED=['common.py','world_up.py','avp.py','inpaint.py','rank_inpainting.py','sam_foreground.py',
        'train_object.py','object_model.py','diffusion.py','camera_sampling.py','provenance.py','compose.py']


def prepare(case):
    CODE.mkdir(parents=True,exist_ok=True)
    for name in REUSED:
        if not (CODE/name).exists():shutil.copy2(SOURCE/name,CODE/name)
    # Export actual pre-update initialization without changing training or RNG.
    p=CODE/'train_object.py';text=p.read_text()
    needle='    for step in range(first, total):'
    if 'native_initial.png' not in text:
        text=text.replace(needle,'''    with torch.no_grad():
        save_image(dest / "native_initial.png", render(reference, obj.data(), background=torch.ones(3, device="cuda"), aux=False)["rgb"])
    save_json(dest / "native_initial.json", {"completed_steps": first if args.resume else (cfg["coarse"]["iterations"] if args.stage == "refine" else 0), "stage": args.stage})
'''+needle)
        p.write_text(text)
    out=BASE/'gsle_native'/case
    for name in ['records','logs','localization']:(out/name).mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():return
    cfg=json.loads((SOURCE/'config.json').read_text())
    cfg.pop('reference_image',None);cfg.pop('replacement_region',None)
    row=next(r for r in json.loads((BASE/'prompts.json').read_text())['cases'] if r['case']==case)
    cfg.update(target_text=row['A'],replacement_text=row['B'],negative_text='',foreground_prompt_mode='generic_box_clip')
    cfg['inpainting']['negative_text']='multiple separate replacement objects, disassembled machinery, floating object, disconnected parts'
    # Resolve the published ambiguity BEFORE seeing case outcomes: literal Alg.1.
    cfg['avp']['scoring']='literal_supplement_algorithm1'
    cfg['native_localization']=dict(semantic_threshold=0.5,opacity_threshold=0.1,minimum_component_points=100,
          voxel_scene_fraction=0.002,voxel_scale_multiplier=2.0,bbox_quantiles=[0.005,0.995],bbox_margin=0.05)
    cfg['assumptions']=[
      'Independent GSLE paper reproduction; not author-released source.',
      'LangSplat trained fields and official text-vs-generic-negative relevance are reused. Voxel-connected component ranking by normalized positive semantic evidence is a disclosed automatic instance-selection adapter; no reference image/SAM prototype is read.',
      'Generation region is the selected localization bounding box, with only the recorded 5% localization margin. No upright height expansion.',
      'AVP uses literal Supplement Algorithm 1: abs(min(S-1)), argmin across eligible views. Main-paper directional-contrast score is retained as an unselected diagnostic because the publications disagree.',
      'AVP geometry-only orbit adaptations may change distance/elevation to avoid occlusion; no Ours camera is used.',
      'Coarse 6500 steps follows supplement section 1.3 (section 1.2 also mentions 3500); refinement 2000. Effective batches 6 and 4 retained with gradient accumulation.',
      'Existing sphere initialization, GSGen compactness, RGB/mask schedules, Stable Zero123 SDS and DI-SDS Eq5-8 are unchanged. Frozen diffusion/background. DI-SDS unconditional uses the existing drop_all_eq7 padding interpretation.',
      'Unpublished initialization, camera sampling and density constants remain the recorded existing reproduction settings. Three fixed inpainting seeds and text-only CLIP ranking plus generic box SAM are retained industrial adapters.',
      'World up/floor protection uses language-field floor relevance and original scene geometry; this is an explicit industrial adapter.'
    ]
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'prompts.json').write_text(json.dumps({k:row[k] for k in ['A','B']},indent=2))
    pins={name:dict(source_sha256=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),native_sha256=hashlib.sha256((CODE/name).read_bytes()).hexdigest()) for name in REUSED}
    (out/'records/reused_sources.json').write_text(json.dumps(pins,indent=2))
    (out/'records/training_exports.patch').write_text(''.join(difflib.unified_diff((SOURCE/'train_object.py').read_text().splitlines(True),(CODE/'train_object.py').read_text().splitlines(True),fromfile='existing/train_object.py',tofile='native/train_object.py')))
    print('Prepared',case,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');prepare(p.parse_args().case)
