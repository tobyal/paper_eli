"""Localization-only harness: unchanged native GE / DGE semantic tracing.

No target photo, mask, spatial prior, editing prompt or generation model is input.
Runs only missing text queries for F17. M1-03 and M1-06 intentionally share one
text-only prediction because their Text A is identical. No GT-driven selection.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(os.environ.get('ROOT', str(Path(__file__).resolve().parents[3])))
BASE = ROOT / 'baseline_comparison'
OUT = ROOT / 'industrial_swap_v2/outputs/paper_figures_v4/F17_localization_comparison'
QUERIES = {'03': 'a gray portable industrial air cooler', '05': 'a gray wheeled assembly stand'}

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2))

def run_ge(out, prompt):
    template = json.loads((BASE/'gaussianeditor_native/case04/config.json').read_text())
    template.update(case='F17_'+out.name)
    write(out/'config.json', template); write(out/'prompts.json', {'A': prompt})
    for name in ['records','logs','localization','add']: (out/name).mkdir(exist_ok=True)
    os.environ['GE_NATIVE_TRACE_DIR'] = str(out/'localization')
    source = BASE/'native_gaussianeditor/stage.py'
    code = source.read_text()
    old = "out = BASE / 'gaussianeditor_native' / args.case"
    assert code.count(old) == 1
    code = code.replace(old, "out = Path(os.environ['F17_NATIVE_OUTPUT'])")
    # Keep headless native setup and w.update_mask verbatim. Stop before Add-view
    # ranking (irrelevant to localization), accept a genuinely empty selection.
    start = code.index("        assert len(indices) > 0, 'Native text tracing returned no selected Gaussian'")
    end = code.index("        del w.text_segmentor", start)
    code = code[:start] + '''        np.save(out / 'localization/selected_indices.npy', indices)
        write(out / 'records/segmentation_candidates.json', audited)
        write(out / 'records/localization.json', dict(selected_gaussians=len(indices),
            threshold=cfg['segmentation_threshold'], source_scene=cfg['source_scene'],
            prompt=prompts['A'], total_cameras=len(w.colmap_cameras),
            sampled_cameras=[camera_record(w.colmap_cameras[i]) for i in native_order],
            policy=cfg['instance_policy'], stage='localization_only'))
        return
''' + code[end:]
    write(out/'records/harness.json', {'native_adapter': str(source),
        'adapter_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'adaptation': 'output directory and early return after native update_mask; no selection change'})
    sys.argv = [str(source), out.name, 'localize']
    exec(compile(code, str(source), 'exec'), {'__name__':'__main__','__file__':str(source)})

def run_dge(out, prompt):
    code = BASE/'vendor/DGE_native'
    sys.path[:0] = [str(code),str(BASE/'native_dge'),str(ROOT/'scripts')]
    os.chdir(code)
    os.environ['DGE_NATIVE_OUTPUT'] = str(out)
    import numpy as np
    import torch
    from argparse import ArgumentParser
    from gaussiansplatting.scene import GaussianModel
    from gaussiansplatting.scene.camera_scene import CamScene
    from gaussiansplatting.arguments import PipelineParams
    from gaussiansplatting.gaussian_renderer import render
    from threestudio.systems.DGE import DGE
    from threestudio.utils.sam import LangSAMTextSegmentor
    import native_audit
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    for name in ['records','localization']: (out/name).mkdir(exist_ok=True)
    cfg = NS(seg_prompt=prompt, mask_thres=0.8, cache_overwrite=False)
    scene = CamScene(str(ROOT/'data/industrial'), h=512, w=512)
    view_list = random.Random(0).sample(range(len(scene.cameras)), min(20,len(scene.cameras)))
    g = GaussianModel(sh_degree=0, anchor_weight_init_g0=1., anchor_weight_init=.1, anchor_weight_multiplier=2)
    source = ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    g.load_ply(str(source)); g.max_radii2D = torch.zeros(len(g.get_xyz), device='cuda')
    s = NS(cfg=cfg, gaussian=g, pipe=PipelineParams(ArgumentParser()),
        background_tensor=torch.zeros(3,device='cuda'), cache_dir=str(out/'native_cache'),
        trainer=NS(datamodule=NS(train_dataset=NS(scene=scene))),
        view_list=view_list, view_num=len(view_list), origin_frames={})
    # Same PNG quantization as official render_all_view; only selected native
    # tracing views are loaded, since others cannot affect update_mask.
    with torch.no_grad():
        for i in view_list:
            rgb=render(scene.cameras[i],g,s.pipe,s.background_tensor)['render'].permute(1,2,0)
            s.origin_frames[i]=(rgb.clamp(0,1)*255).to(torch.uint8).to(torch.float32)[None]/255
    s.text_segmentor=LangSAMTextSegmentor().to('cuda')
    native_audit.install(s)
    write(out/'config.json',dict(prompt=prompt,source_scene=str(source),seed=0,resolution=512,
        segmentation_views=20,mask_threshold=.8,instance_policy='unchanged first LangSAM candidate',
        sampled_cameras=[scene.cameras[i].image_name for i in view_list],
        localization_only=True,editing_or_diffusion_initialized=False))
    # Exact vendored native function and auditing hooks; no Lightning trainer,
    # on_fit_start, edit_all_view, diffusion guidance or optimizer steps.
    DGE.update_mask(s)
    write(out/'records/harness.json',{'source':str(code/'threestudio/systems/DGE.py'),
        'source_sha256':hashlib.sha256((code/'threestudio/systems/DGE.py').read_bytes()).hexdigest(),
        'entry':'DGE.update_mask', 'adaptation':'localization-only namespace, native sampled views and PNG-quantized renders'})

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['GaussianEditor','DGE']);p.add_argument('query',choices=QUERIES);p.add_argument('--gpu',default='2');a=p.parse_args()
    out=OUT/'native_localization'/a.method/('query_'+a.query);out.mkdir(parents=True,exist_ok=True)
    done=out/'complete.json'
    if done.exists(): print('Already completed',out);return
    os.environ.update(ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpu,F17_NATIVE_OUTPUT=str(out),
        HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1')
    for key in ['TARGET_GEOMETRY','INSTANCE_SELECTION_LOG','GE_TARGET_CAMERA_DEPTH','GE_REUSE_ADD_ASSETS']:
        os.environ.pop(key,None)
    start=time.time()
    (run_ge if a.method=='GaussianEditor' else run_dge)(out,QUERIES[a.query])
    write(done,{'method':a.method,'prompt':QUERIES[a.query],'seconds':time.time()-start,
        'editing_steps':0,'model_training_steps':0,'photo_or_GT_inputs':False})
    print('Completed localization only',a.method,a.query,flush=True)

if __name__=='__main__': main()
