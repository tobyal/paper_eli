"""M2: actual scene conditions, one Qwen image, automatically segmented foreground."""
import subprocess
from PIL import Image
from ..core.context import Context, arguments, read_json, write_json
from ..core.depth import prepare_conditions
from ..core.qwen_client import generate
from ..core.cameras import resized_camera


def build_prompt(ctx):
    return (
        "Edit image 1 and return one photorealistic RGB image of the same factory from exactly the same camera. "
        "Image 1 is the original scene. Image 2 is ONLY a position guide: the translucent magenta polygon encloses the allowed replacement volume; remove all magenta markings in the output. "
        "Replace the entire old machine consisting of a black assembly and a white support base inside that polygon. "
        "Image 3 is ONLY a background depth reference, with the editing area blacked out as UNKNOWN depth. Use it for surrounding floor/equipment perspective, not the old machine's shape. Do not copy its grayscale appearance. "
        "Generate exactly one complete device: " + ctx.cfg["m2"]["replacement_text"] + " "
        "Show a coherent complete base, shoulder, elbow, wrist, distinct rotary joints and gripper with clear solid mechanical geometry. "
        "Place its base on the factory floor at the old device's base location. Fit the complete arm, gripper and pedestal within the marked volume without clipping. "
        "Keep the original camera, surrounding equipment, blue containers, railings, floor and lighting unchanged. Match the factory perspective, scale, color temperature and shading. "
        "Generate no cabinet, enclosure, extra machinery, duplicate arm, text, annotations or new background behind the robot. The output is the edited full factory view, not a collage or depth map."
    )


def run(ctx, prepare_only=False):
    out = ctx.stage("m2")
    if not (out / "conditions.json").exists():
        prepare_conditions(ctx)
    images = [out / "anchor.png", out / "position_hint.png", out / "depth_background.png"]
    prompt = build_prompt(ctx)
    write_json(out / "generation_plan.json", {"model": ctx.cfg["m2"]["model"], "images": list(map(str, images)), "prompt": prompt, "max_requests": 1, "candidates": 1})
    (out / 'generation_prompt.txt').write_text(prompt, encoding='utf-8')
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
