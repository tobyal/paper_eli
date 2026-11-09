"""Headless native UI automation with stage images and explicit input provenance.

The upstream tracing/Delete/Add kernels run unchanged apart from cache, runtime
compatibility and audit exports. No Ours spatial file is accepted by this entry.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'
CODE = BASE / 'vendor/GaussianEditor_native'


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


class DisplayHandle:
    def remove(self):
        pass

    def on_click(self, callback):
        return callback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('stage', choices=['localize', 'delete', 'add', 'export'])
    args = p.parse_args()
    out = BASE / 'gaussianeditor_native' / args.case
    cfg = json.loads((out / 'config.json').read_text())
    prompts = json.loads((out / 'prompts.json').read_text())
    for key in ['TARGET_GEOMETRY', 'INSTANCE_SELECTION_LOG', 'GE_TARGET_CAMERA_DEPTH', 'GE_REUSE_ADD_ASSETS']:
        assert not os.environ.get(key), 'Forbidden inherited spatial/cache input: ' + key
    os.environ.update(CUDA_HOME='/usr/local/cuda-11.8',
                      TORCH_EXTENSIONS_DIR=str(BASE / 'cache/torch_extensions'),
                      TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4')
    os.environ['PATH'] = '/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', '')
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from torchvision.transforms.functional import to_pil_image
    from plyfile import PlyData, PlyElement
    import webui
    import ui_utils
    from webui import WebUI, GaussianModel, CamScene, PipelineParams, LangSAMTextSegmentor

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    ui_utils.new_frustums = lambda *a, **kw: DisplayHandle()
    ui_utils.new_frustum_from_cam = lambda *a, **kw: (DisplayHandle(), DisplayHandle())
    w = WebUI.__new__(WebUI)
    for node in ast.walk(ast.parse((CODE / 'webui.py').read_text())):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                    for kw in node.value.keywords:
                        if kw.arg == 'initial_value':
                            try:
                                setattr(w, target.attr, SimpleNamespace(value=ast.literal_eval(kw.value), visible=False))
                            except (ValueError, TypeError):
                                pass
    for name in ['seg_scale_end_button', 'inpaint_end', 'depth_end']:
        setattr(w, name, SimpleNamespace(value=False, visible=False))
    w.server = SimpleNamespace(get_clients=lambda: {0: SimpleNamespace(camera=None)},
                               add_frame=lambda *a, **kw: DisplayHandle(),
                               add_camera_frustum=lambda *a, **kw: DisplayHandle())
    w.use_sam = False
    w.guidance = None
    w.stop_training = False
    w.ctn_inpaint = None
    w.ip2p = None
    w.points3d = []
    w.seg_scale = w.seg_scale_end = True
    w.inpaint_again = w.inpaint_end_flag = True
    w.scale_depth = w.depth_end_flag = True
    w.seg_cam_num.value = cfg['segmentation_views']
    w.mask_thres.value = cfg['segmentation_threshold']
    w.edit_cam_num.value = cfg['delete_views']
    w.edit_train_steps.value = cfg['delete_steps']
    # This is the upstream Edit Type=Delete callback, absent in older adapters.
    w.densification_interval.value = cfg['delete_densification_interval']
    w.depth_scaler.value = cfg['depth_scale']
    assert 0 < w.depth_scaler.value <= 5, 'Native GUI Depth Scale range'
    w.pipe = PipelineParams(argparse.ArgumentParser())
    w.background_tensor = torch.zeros(3, device='cuda')
    scene = CamScene(cfg['colmap_source'], h=cfg['resolution'], w=cfg['resolution'])
    w.colmap_cameras = scene.cameras
    w.cameras_extent = scene.cameras_extent

    def load_model(path):
        g = GaussianModel(sh_degree=0, anchor_weight_init_g0=w.anchor_weight_init_g0.value,
                          anchor_weight_init=w.anchor_weight_init.value,
                          anchor_weight_multiplier=w.anchor_weight_multiplier.value)
        g.load_ply(str(path))
        g.max_radii2D = torch.zeros(len(g.get_xyz), device='cuda')
        return g

    source = cfg['source_scene'] if args.stage in ['localize', 'delete'] else str(out / 'scene_deleted.ply')
    if args.stage == 'export':
        source = str(out / 'final_scene.ply')
    w.gaussian = load_model(source)
    if args.stage != 'export':
        w.configure_optimizers()
        w.gaussian.update_anchor_term(w.anchor_weight_init_g0.value, w.anchor_weight_init.value,
                                     w.anchor_weight_multiplier.value)
    write(out / 'records' / (args.stage + '_controls.json'),
          {k: v.value for k, v in vars(w).items() if isinstance(v, SimpleNamespace) and hasattr(v, 'value')})

    def rgb(cam, model=None):
        with torch.no_grad():
            return webui.render(cam, model or w.gaussian, w.pipe, w.background_tensor)['render'].clamp(0, 1)

    def pil(tensor):
        return to_pil_image(tensor.detach().cpu().clamp(0, 1))

    def camera_record(cam):
        return dict(name=cam.image_name, R=cam.R.tolist(), T=cam.T.tolist(), FoVx=cam.FoVx,
                    FoVy=cam.FoVy, width=cam.image_width, height=cam.image_height)

    def bounds(mask):
        yy, xx = np.where(mask)
        return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if len(xx) else None

    def overlay(image, mask=None, box=None, title=None):
        image = image.convert('RGB')
        if mask is not None:
            array = np.asarray(image).copy()
            array[mask] = (array[mask].astype(float) * 0.55 + np.array([255, 30, 30]) * 0.45).astype(np.uint8)
            image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        if box:
            draw.rectangle(box, outline=(255, 190, 0), width=3)
        if title:
            draw.rectangle((0, 0, image.width, 22), fill='white')
            draw.text((5, 5), title, fill='black')
        return image

    def contact(items, path, columns=4):
        canvas = Image.new('RGB', (320 * columns, 342 * ((len(items) + columns - 1) // columns)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (name, im) in enumerate(items):
            x, y = (i % columns) * 320, (i // columns) * 342
            draw.text((x + 5, y + 5), name, fill='black')
            canvas.paste(ImageOps.contain(im, (320, 320)), (x, y + 22))
        canvas.save(path)

    if args.stage == 'localize':
        native_order = random.Random(0).sample(range(len(w.colmap_cameras)), min(len(w.colmap_cameras), w.seg_cam_num.value))
        w.text_segmentor = LangSAMTextSegmentor().to('cuda')
        original_predict = w.text_segmentor.model.predict
        audited = []

        def audit_predict(image, prompt, *a, **kw):
            result = original_predict(image, prompt, *a, **kw)
            masks, boxes, phrases, logits = result
            cam = w.colmap_cameras[native_order[len(audited)]]
            folder = out / 'localization' / cam.image_name
            folder.mkdir(exist_ok=True)
            image.save(folder / 'input.png')
            count = len(masks) if masks.ndim == 3 else 0
            record = dict(camera=cam.image_name, candidate_count=count, selected_index=0 if count else None,
                          policy='unchanged upstream first candidate', boxes=boxes.tolist(),
                          phrases=phrases, scores=logits.tolist())
            for i in range(count):
                m = masks[i].detach().cpu().numpy().astype(bool)
                Image.fromarray((m * 255).astype(np.uint8)).save(folder / ('candidate_%02d_mask.png' % i))
                overlay(image, m, bounds(m)).save(folder / ('candidate_%02d_overlay.png' % i))
            write(folder / 'candidates.json', record)
            audited.append(record)
            return result

        w.text_segmentor.model.predict = audit_predict
        _, selected = w.update_mask(prompts['A'])
        indices = selected.nonzero().flatten().cpu().numpy()
        assert len(indices) > 0, 'Native text tracing returned no selected Gaussian'
        np.save(out / 'localization/selected_indices.npy', indices)
        write(out / 'records/segmentation_candidates.json', audited)
        del w.text_segmentor
        torch.cuda.empty_cache()
        # Actual contribution from GE-selected Gaussians; no projected Ours box.
        ranking = []
        colors = selected[:, None].float().repeat(1, 3)
        with torch.no_grad():
            for i, cam in enumerate(w.colmap_cameras):
                contribution = webui.render(cam, w.gaussian, w.pipe, w.background_tensor, override_color=colors)['render'][0]
                mask = (contribution > cfg['visible_mask_threshold']).cpu().numpy()
                box = bounds(mask)
                if not box:
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]
                margin = cfg['add_margin_fraction'] * np.array([bw, bh])
                lo = np.floor(np.array(box[:2]) - margin).astype(int)
                hi = np.ceil(np.array(box[2:]) + margin).astype(int)
                border = cfg['border_fraction'] * np.array([cam.image_width, cam.image_height])
                framed = bool((lo >= border).all() and (hi <= np.array([cam.image_width, cam.image_height]) - border).all())
                ranking.append(dict(camera_index=i, camera=cam.image_name, pixels=int(mask.sum()),
                                    bbox=box, add_bbox=[int(v) for v in [*lo, *hi]], fully_framed=framed))
        ranking.sort(key=lambda x: (x['fully_framed'], x['pixels']), reverse=True)
        write(out / 'records/camera_ranking.json', ranking)
        assert ranking and ranking[0]['fully_framed'], 'No fully framed GE target; inspect native semantic results'
        chosen = ranking[0]
        cam = w.colmap_cameras[chosen['camera_index']]
        write(out / 'anchor_camera.json', camera_record(cam))
        write(out / 'records/add_region.json', dict(source='GE traced-Gaussian rendered visible mask bbox + relative margin',
                                                   camera=cam.image_name, mask_threshold=cfg['visible_mask_threshold'],
                                                   margin_fraction=cfg['add_margin_fraction'], bbox=chosen['add_bbox']))
        image = pil(rgb(cam))
        image.save(out / 'original_view.png')
        with torch.no_grad():
            contribution = webui.render(cam, w.gaussian, w.pipe, w.background_tensor, override_color=colors)['render'][0]
            mask = (contribution > cfg['visible_mask_threshold']).cpu().numpy()
            native_mask = contribution.cpu().numpy() > 0
            w.gaussian.localize = True
            pil(rgb(cam)).save(out / 'localization_render_selected.png')
            w.gaussian.localize = False
        Image.fromarray((native_mask * 255).astype(np.uint8)).save(out / 'delete_target_mask.png')
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / 'localization_visible_mask.png')
        overlay(image, mask).save(out / 'localization_mask_overlay.png')
        overlay(image, None, chosen['bbox']).save(out / 'localization_scene_with_box.png')
        overlay(image, None, chosen['add_bbox']).save(out / 'add_region_overlay.png')
        add_mask = np.zeros(mask.shape, np.uint8)
        x0, y0, x1, y1 = chosen['add_bbox']
        add_mask[y0:y1, x0:x1] = 255
        Image.fromarray(add_mask).save(out / 'add_region_mask.png')
        previews = []
        with torch.no_grad():
            for row in ranking[:12]:
                c = w.colmap_cameras[row['camera_index']]
                contribution = webui.render(c, w.gaussian, w.pipe, w.background_tensor, override_color=colors)['render'][0]
                m = (contribution > cfg['visible_mask_threshold']).cpu().numpy()
                im = overlay(pil(rgb(c)), m, row['bbox'])
                im.save(out / 'localization' / (c.image_name + '_traced.png'))
                previews.append((c.image_name + ' | %d px' % row['pixels'], im))
        contact(previews, out / 'localization_views.png')
        vertices = PlyData.read(cfg['source_scene'])['vertex'].data
        PlyData([PlyElement.describe(vertices[indices], 'vertex')], text=False).write(str(out / 'selected_target.ply'))
        xyz = w.gaussian.get_xyz[selected].detach().cpu().numpy()
        write(out / 'records/localization.json', dict(gaussians=int(len(indices)), scene_gaussians=len(vertices),
              bbox_low=xyz.min(0).tolist(), bbox_high=xyz.max(0).tolist(), selected_camera=cam.image_name,
              selection_policy=cfg['instance_policy'], source_scene=cfg['source_scene'],
              colmap_source=cfg['colmap_source'], total_cameras=len(w.colmap_cameras),
              segmentation_cameras=[w.colmap_cameras[i].image_name for i in native_order],
              source='Only native text masks and Gaussian semantic tracing; no Ours geometry'))
    else:
        anchor = json.loads((out / 'anchor_camera.json').read_text())
        cam = next(c for c in w.colmap_cameras if c.image_name == anchor['name'])
        region = json.loads((out / 'records/add_region.json').read_text())['bbox']
        if args.stage == 'delete':
            selected = torch.zeros(len(w.gaussian.get_xyz), device='cuda', dtype=torch.bool)
            selected[torch.as_tensor(np.load(out / 'localization/selected_indices.npy'), device='cuda')] = True
            w.gaussian.set_mask(selected)
            w.gaussian.apply_grad_mask(selected)
            pil(rgb(cam)).save(out / 'delete_before.png')
            w.edit_text.value = prompts['background_text']
            cameras, frames, frustums = ui_utils.sample_train_camera(w.colmap_cameras, w.edit_cam_num.value, w.server)
            write(out / 'records/delete_cameras.json', [camera_record(c) for c in cameras])
            original_masks = w.render_all_view_with_mask

            def audit_delete_masks(*a, **kw):
                masks, originals = original_masks(*a, **kw)
                folder = out / 'delete_conditions'
                folder.mkdir(exist_ok=True)
                for i, (m, im) in enumerate(zip(masks, originals)):
                    pil(m.squeeze().float()).save(folder / ('%02d_inpaint_mask.png' % i))
                    pil(im[0].permute(2, 0, 1)).save(folder / ('%02d_pruned_scene.png' % i))
                return masks, originals

            w.render_all_view_with_mask = audit_delete_masks
            w.delete(cameras, frames, frustums)
            w.gaussian.anchor_postfix()
            w.gaussian.save_ply(str(out / 'scene_deleted.ply'))
            after = pil(rgb(cam))
            after.save(out / 'delete_after.png')
            overlay(after, None, region).save(out / 'add_region_with_deleted_scene.png')
            contact([('Before', Image.open(out / 'delete_before.png')), ('After native Delete', after)],
                    out / 'delete_comparison.png', columns=2)
            for i, im in w.guidance.edit_frames.items():
                pil(im[0].permute(2, 0, 1)).save(out / 'delete_conditions' / ('%02d_diffusion_target.png' % i))
        elif args.stage == 'add':
            w.left_up.value = region[:2]
            w.right_down.value = region[2:]
            w.edit_text.value = prompts['B']
            w.refine_text.value = prompts['gaussianeditor_refine_text']
            w.inpaint_seed.value = cfg['seed']
            assert not (out / 'add/inpainted.png').exists(), 'Do not silently reuse any previous generated image'
            write(out / 'records/add_inputs.json', dict(camera=camera_record(cam), bbox=region,
                  region_source='records/add_region.json', add_text=prompts['B'],
                  refine_text=prompts['gaussianeditor_refine_text'], depth_scale=cfg['depth_scale']))
            w.add(cam)
            pil(rgb(cam)).save(out / 'scene_inserted_before_refine.png')
            w.gaussian.anchor_postfix()
            w.gaussian.save_ply(str(out / 'final_scene.ply'))
            pil(rgb(cam)).save(out / 'scene_final.png')
            write(out / 'records/stage_semantics.json', {
                'refinement_order':'Native mesh-to-Gaussian/IP2P refinement runs BEFORE scene insertion.',
                'post_insertion_refinement':False,
                'scene_inserted_before_refine.png':'Compatibility filename: actual first inserted scene; no subsequent scene-refine stage exists in native Add.',
                'scene_final.png':'Same native insertion result after bookkeeping anchor_postfix; not an additional optimization.'})
        else:
            shutil.copy2(out / 'add/inpainted.png', out / 'generated_2d_image.png')
            shutil.copy2(out / 'add/removed_bg.png', out / 'generated_2d_rgba.png')
            rgba = Image.open(out / 'generated_2d_rgba.png').convert('RGBA')
            rgba.getchannel('A').save(out / 'generated_2d_mask.png')
            object_model = load_model(out / 'add/object_world.ply')
            pil(rgb(cam, object_model)).save(out / 'generated_object_3d_preview.png')
            final = pil(rgb(cam))
            final.save(out / 'scene_final.png')
            final.crop(tuple(region)).save(out / 'scene_final_crop.png')
            contact([('Original', Image.open(out / 'original_view.png')),
                     ('Native Delete', Image.open(out / 'delete_after.png')), ('Native Add final', final)],
                    out / 'final_comparison.png', columns=3)
            ranking = json.loads((out / 'records/camera_ranking.json').read_text())
            main_center = np.asarray(cam.camera_center.cpu())
            others = sorted(ranking[1:], key=lambda row: float(np.linalg.norm(
                w.colmap_cameras[row['camera_index']].camera_center.cpu().numpy() - main_center)), reverse=True)
            selected_views = [ranking[0]] + others[:2]
            multiview = []
            for i, row in enumerate(selected_views):
                c = w.colmap_cameras[row['camera_index']]
                im = pil(rgb(c))
                im.save(out / ('final_view_%d.png' % i))
                multiview.append((c.image_name, im))
            contact(multiview, out / 'final_multiview.png', columns=3)
            write(out / 'records/final_cameras.json', [camera_record(w.colmap_cameras[x['camera_index']]) for x in selected_views])
    print(args.case, args.stage, 'complete', flush=True)


if __name__ == '__main__':
    main()
