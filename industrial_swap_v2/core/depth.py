"""Render original RGB/depth and mark unknown depth in the editing volume."""
import numpy as np
import torch
from PIL import Image
import cv2
import shutil
from .context import gsle, read_json, write_json
from .cameras import resized_camera


def prepare_conditions(ctx):
    c = gsle(ctx)
    out = ctx.stage("m2")
    region = read_json(ctx.stage("m1") / "edit_region.json")
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
        avp.main()
        anchor_path=work/'avp/anchor.json'
    anchor = read_json(anchor_path)
    resolution = ctx.cfg["m2"]["anchor_resolution"]
    camera = c.Camera.from_record(anchor["camera"]).resize(resolution, resolution)
    scene = c.GaussianData.load(ctx.source(ctx.cfg["project"]["source_ply"]))
    with torch.no_grad():
        result = c.render(camera, scene)
        mask = c.bbox_mask(camera, np.asarray(region["bbox_corners"]))
        depth = c.depth_condition(result["depth"], result["alpha"])
        c.save_image(out / "anchor.png", result["rgb"])
        c.save_image(out / "depth.png", depth)
        c.save_image(out / "depth_background.png", depth * (1 - torch.as_tensor(mask, device="cuda")))
        c.save_image(out / "edit_mask.png", mask)
        np.savez_compressed(out / "depth.npz", depth=result["depth"].cpu().numpy(), alpha=result["alpha"].cpu().numpy(), edit_mask=mask)
    rgb = np.asarray(Image.open(out / "anchor.png")).copy()
    marked = rgb.copy()
    selected = mask.astype(bool)
    marked[selected] = (0.8 * rgb[selected] + 0.2 * np.array([255, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (255, 0, 255), max(2, resolution // 256))
    Image.fromarray(marked).save(out / "position_hint.png")
    write_json(out / "anchor_camera.json", camera.record())
    write_json(out / "conditions.json", {"anchor_reused": ctx.cfg['m2']['reuse_anchor'], "source_anchor": str(anchor_path),
               "depth_definition": "GSLE volumetrically weighted Euclidean distance; minmax then inverse",
               "background_depth": "same original depth, zeroed inside projected generation box; zero means unknown here",
               "camera": camera.record(), "mask_area": int(mask.sum())})
    del scene, result, depth
    torch.cuda.empty_cache()
    return out
