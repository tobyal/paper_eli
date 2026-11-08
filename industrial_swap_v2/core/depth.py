"""M2 adaptive generation space and visibility-constrained image conditions."""
import numpy as np
import torch
from PIL import Image
import cv2
import shutil
from .context import gsle, read_json, write_json, sha256
from .cameras import resized_camera
from .m2_space import VERSION, settings, signature, build_generation_region
from .m2_edit_mask import render_geometry, build_mask
from .scene import read_vertices


def prepare_conditions(ctx):
    c = gsle(ctx)
    out = ctx.stage("m2")
    source_path=ctx.source(ctx.cfg['project']['source_ply'])
    source_hash=sha256(source_path)
    if source_hash!=ctx.cfg['project']['source_sha256']:
        raise ValueError('Original scene SHA256 differs from the confirmed M1 source')
    if (out/'conditions.json').exists():
        raise FileExistsError('Prepare adaptive conditions in a new experiment directory')
    m1 = ctx.stage('m1')
    input_names=('edit_region.json','target.json','target_indices.npy','target_gaussians.ply','background_gaussians.ply')
    input_hashes={name:sha256(m1/name) for name in input_names}
    target = read_json(m1/'target.json')
    cfg = settings(ctx.cfg)
    region = build_generation_region(read_vertices(m1/'target_gaussians.ply'),target,
        ctx.cfg['m2']['replacement_text'],cfg)
    write_json(out/'generation_region.json',region)
    anchor_path = ctx.source(ctx.cfg['m2']['existing_anchor'])
    if not ctx.cfg['m2']['reuse_anchor']:
        import avp
        work = out/'avp_work'
        (work/'localization').mkdir(parents=True,exist_ok=True)
        shutil.copy2(ctx.stage('m1')/'target_indices.npy',work/'localization/target_indices.npy')
        avp.OUT = work
        avp.editing_region = lambda: region
        # Historical AVP has one hard-coded record destination. Redirect that
        # write in this adapter, preserving the old source and old artifacts.
        def safe_save(path, data):
            if path == ctx.root/'records/gsle/avp.json':
                path = ctx.records/'avp.json'
            path.resolve().relative_to(ctx.base.resolve())
            write_json(path,data)
        avp.save_json = safe_save
        try:
            avp.main()
        except RuntimeError as error:
            if not ctx.cfg['m2'].get('avp_observed_fallback') or 'No visible AVP candidate' not in str(error):
                raise
            # Existing project fallback: unchanged brightness/visibility rules,
            # calibrated observed candidates instead of an unsupported orbit.
            from .observed_anchor import select_observed_anchor
            try:
                select_observed_anchor(c,work,region,np.load(ctx.stage('m1')/'target_indices.npy'),
                                       avp.config()['avp'],avp.brightness_scores)
            except RuntimeError as observed_error:
                if not ctx.cfg['m2'].get('avp_review_fallback') or 'Neither orbit nor observed' not in str(observed_error):
                    raise
                select_observed_anchor(c,work,region,np.load(ctx.stage('m1')/'target_indices.npy'),
                                       avp.config()['avp'],avp.brightness_scores,expand_canvas=True,allow_review=True)
        anchor_path=work/'avp/anchor.json'
    anchor = read_json(anchor_path)
    resolution = ctx.cfg["m2"]["anchor_resolution"]
    camera = c.Camera.from_record(anchor.get('camera',anchor)).resize(resolution, resolution)
    scene = c.GaussianData.load(ctx.source(ctx.cfg["project"]["source_ply"]))
    with torch.no_grad():
        result = c.render(camera, scene)
        box_mask = c.bbox_mask(camera, np.asarray(region["bbox_corners"])).astype(bool)
        if not box_mask.any():
            raise ValueError('Generation volume is outside the Anchor; select a new AVP view')
        maps = render_geometry(c,camera,scene,np.load(m1/'target_indices.npy'),region,cfg)
        mask,diagnostics,mask_report = build_mask(camera,maps,region,box_mask,cfg)
        mask=mask.astype(np.uint8)
        depth = c.depth_condition(result["depth"], result["alpha"])
        c.save_image(out / "anchor.png", result["rgb"])
        c.save_image(out / "depth.png", depth)
        c.save_image(out / "depth_background.png", depth * (1 - torch.as_tensor(mask, device="cuda")))
        c.save_image(out / "edit_mask.png", mask)
        np.savez_compressed(out / "depth.npz", depth=result["depth"].cpu().numpy(), alpha=result["alpha"].cpu().numpy(), edit_mask=mask)
        for name,image in diagnostics.items():
            c.save_image(out/(name+'.png'),image.astype(np.uint8))
        np.savez_compressed(out/'spatial_depth.npz',**maps)
    rgb = np.asarray(Image.open(out / "anchor.png")).copy()
    marked = rgb.copy()
    selected = mask.astype(bool)
    marked[selected] = (0.8 * rgb[selected] + 0.2 * np.array([255, 0, 255])).astype(np.uint8)
    width=max(2,resolution//256)
    inner=cv2.erode(mask,np.ones((2*width+1,2*width+1),np.uint8)).astype(bool)
    marked[selected&~inner]=(255,0,255)
    Image.fromarray(marked).save(out / "position_hint.png")
    write_json(out/'mask_report.json',mask_report)
    write_json(out / "anchor_camera.json", camera.record())
    write_json(out / "conditions.json", {"anchor_reused": ctx.cfg['m2']['reuse_anchor'], "source_anchor": str(anchor_path),
               'spatial_method':VERSION,'condition_signature':signature(ctx.cfg),
               'generation_region_file':'generation_region.json','generation_region_sha256':sha256(out/'generation_region.json'),
               'semantic_profile':region['semantic_profile'],'mask_report':mask_report,
               "depth_definition": "GSLE volumetrically weighted Euclidean distance; minmax then inverse",
               'geometry_depth_definition':'camera-z moments plus Gaussian transmittance-weighted depth-prefix CDF; foreground guards query mass before/inside the replacement volume',
               "background_depth": "same original depth, zeroed only inside visible-target-plus-depth-guarded-extension mask; zero means unknown here",
               "camera": camera.record(), "mask_area": int(mask.sum()),
               "m1_reference_camera":read_json(ctx.stage('m1')/'target.json')['reference_camera'],
               "editing_region":region['bbox_corners'],
               "avp_selection":{k:anchor[k] for k in ('scoring','eligible_views','selected','candidate_pool','adaptation','quality_status','quality_note','thresholds') if k in anchor},
               "m1_case_id":ctx.cfg['m2'].get('confirmed_m1_case_id'),
               "source_scene_sha256":source_hash,
               "input_sha256":input_hashes})
    assert input_hashes=={name:sha256(m1/name) for name in input_names}, 'M1 changed during preparation'
    del scene, result, depth
    torch.cuda.empty_cache()
    return out
