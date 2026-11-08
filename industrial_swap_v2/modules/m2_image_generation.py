"""M2: actual scene conditions, one Qwen image, automatically segmented foreground."""
import subprocess
from PIL import Image
from ..core.context import Context, arguments, read_json, write_json
from ..core.depth import prepare_conditions
from ..core.qwen_client import generate
from ..core.cameras import resized_camera
from ..core.m2_space import VERSION, signature


from ..core.m2_prompt import build_prompt


def run(ctx, prepare_only=False):
    out = ctx.stage("m2")
    if not (out / "conditions.json").exists():
        prepare_conditions(ctx)
    else:
        saved=read_json(out/'conditions.json')
        if saved.get('spatial_method')!=VERSION or saved.get('condition_signature')!=signature(ctx.cfg):
            raise ValueError('M2 description or spatial method changed; prepare a new experiment version instead of overwriting existing conditions')
    images = [out / "anchor.png", out / "position_hint.png", out / "depth_background.png"]
    prompt = build_prompt(ctx)
    write_json(out / "generation_plan.json", {"model": ctx.cfg["m2"]["model"], "images": list(map(str, images)), "prompt": prompt, "max_requests": 1, "candidates": 1})
    (out / 'generation_prompt.txt').write_text(prompt, encoding='utf-8')
    write_json(out/'generation_inputs.json',{'model':ctx.cfg['m2']['model'],
        'image_order':['anchor.png','position_hint.png','depth_background.png'],
        'inspection_only':['depth.png','edit_mask.png','depth.npz','generation_region.json','mask_report.json'],
        'prompt_file':'generation_prompt.txt','camera_file':'anchor_camera.json',
        'm1_case_id':ctx.cfg['m2'].get('confirmed_m1_case_id'),
        'replacement_text':ctx.cfg['m2']['replacement_text'],
        'stage':'prepared_inputs_only' if prepare_only else 'generation_requested',
        'background_depth_zero_means':'unknown only inside visibility-constrained edit_mask.png'})
    if prepare_only:
        ctx.receipt("m2", "conditions_prepared_api_not_called", images=list(map(str, images)))
        print("M2 conditions ready; Qwen has not been called", flush=True)
        return
    generate(ctx, images, prompt)
    with Image.open(out / "edited_scene.png") as image:
        width, height = image.size
    original_camera = read_json(out / "conditions.json")["camera"]
    assert abs(width / height - original_camera["width"] / original_camera["height"]) < 1e-6, "Qwen changed image aspect ratio"
    write_json(out / "anchor_camera.json", resized_camera(original_camera, width, height))
    subprocess.run([str(ctx.source(ctx.cfg["project"]["sam_python"])), "-m", "industrial_swap_v2.core.segmentation", "--config", str(ctx.config_path)], cwd=ctx.root, env=ctx.env(), check=True)
    ctx.complete("m2", [out / n for n in ("anchor.png", "edit_mask.png", "depth.png", "depth_background.png", "edited_scene.png", "object_rgba.png", "object_mask.png", "object_white.png", "anchor_camera.json", "generation.json")],
                 successful_generation_requests=1, submission_attempts=1+len(list(ctx.records.glob('qwen_attempt_*.lock'))))
    print("M2 complete", flush=True)


if __name__ == "__main__":
    args = arguments()
    run(Context(args.config), args.prepare_only)
