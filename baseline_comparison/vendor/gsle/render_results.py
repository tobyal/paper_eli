"""Actual before/after images, calibrated video path, and independent 360 views."""

import argparse, math, subprocess, time
import cv2
from PIL import Image, ImageDraw
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["coarse", "refine"])
parser.add_argument("--preview", action="store_true")
args = parser.parse_args()
folder = "composition" if args.stage == "coarse" else "final"
dest = OUT / folder / "renders"
dest.mkdir(parents=True, exist_ok=True)
scene = GaussianData.load(OUT / folder / "scene.ply")
obj = GaussianData.load(OUT / folder / "object_world.ply")
source = GaussianData.load(
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
target = editing_region()
anchor = json.loads((OUT / "avp/anchor.json").read_text())
old_ids = np.load(OUT / "localization/target_indices.npy")
old_object = source.subset(old_ids)
cameras = source_cameras(960)
if args.preview:
    cameras = [
        c
        for i, c in enumerate(cameras)
        if i % 35 == 10 or c.name == config()["reference_image"]
    ]
metrics = []
start = time.time()
with torch.no_grad():
    for i, cam in enumerate(cameras):
        before = render(cam, source, aux=False)["rgb"]
        after = render(cam, scene, aux=False)["rgb"]
        old_alpha = render(cam, old_object)["alpha"]
        new_alpha = render(cam, obj)["alpha"]
        edit = ((old_alpha[0] + new_alpha[0]) > 0.01).cpu().numpy().astype(np.uint8)
        edit = cv2.dilate(edit, np.ones((11, 11), np.uint8))
        outside = torch.as_tensor(edit == 0, device="cuda")
        background_mae = (
            float((before - after).abs()[:, outside].mean()) if outside.any() else None
        )
        stem = Path(cam.name).stem
        save_image(dest / f"before/{stem}.png", before)
        save_image(dest / f"after/{stem}.png", after)
        side = torch.cat((before, after), dim=2)
        save_image(dest / f"side_by_side/{stem}.png", side)
        metrics.append(
            {
                "camera": cam.name,
                "background_mae_outside_old_new_union": background_mae,
                "edit_pixel_fraction": float(edit.mean()),
                "camera_record": cam.record(),
            }
        )
        if i % 25 == 0:
            print("Render", i + 1, len(cameras), flush=True)
    # Independent orbit around the object, not frames copied from the input video.
    reference = Camera.from_record(anchor["camera"])
    center = np.asarray(target["center"])
    up = np.asarray(target["up"])
    offset = reference.center - center
    radius = np.linalg.norm(offset)
    horizontal = normalized(offset - up * np.dot(offset, up))
    right = normalized(np.cross(up, horizontal))
    elevation = math.radians(anchor["selected"]["elevation"])
    count = 12 if args.preview else 72
    for i in range(count):
        angle = 2 * math.pi * i / count
        position = center + radius * (
            math.cos(elevation)
            * (math.cos(angle) * horizontal + math.sin(angle) * right)
            + math.sin(elevation) * up
        )
        cam = look_at(position, center, up, 512, 49.1)
        save_image(
            dest / f"orbit_object/{i:03d}.png",
            render(cam, obj, background=torch.ones(3, device="cuda"), aux=False)["rgb"],
        )
        save_image(
            dest / f"orbit_scene/{i:03d}.png", render(cam, scene, aux=False)["rgb"]
        )
save_json(
    dest / "render_metrics.json",
    {
        "stage": args.stage,
        "preview_subset": args.preview,
        "views": metrics,
        "elapsed": time.time() - start,
        "trajectory_source": "original calibrated cameras, extracted at2fps",
        "source_video_duration": 174.89655,
    },
)


def encode(pattern, output, fps):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-threads",
            "8",
            "-crf",
            "18",
            "-g",
            str(2 * fps),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


if not args.preview:
    encode(dest / "side_by_side/*.png", dest / "comparison_original_speed_2fps.mp4", 2)
    encode(dest / "side_by_side/*.png", dest / "comparison_6x_preview_12fps.mp4", 12)
    encode(dest / "after/*.png", dest / "edited_original_speed_2fps.mp4", 2)
    encode(dest / "orbit_object/*.png", dest / "object_360.mp4", 12)
    encode(dest / "orbit_scene/*.png", dest / "scene_360.mp4", 12)
files = sorted((dest / "orbit_object").glob("*.png"))
chosen = [files[i] for i in np.linspace(0, len(files) - 1, 12).astype(int)]
sheet = Image.new("RGB", (4 * 256, 3 * 256), "white")
for i, path in enumerate(chosen):
    sheet.paste(Image.open(path).resize((256, 256)), ((i % 4) * 256, (i // 4) * 256))
sheet.save(dest / "object_contact_sheet.jpg")
print("Actual rendered artifacts:", dest, flush=True)
