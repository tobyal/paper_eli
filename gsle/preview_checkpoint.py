"""Read-only orbit QA of a saved object checkpoint while training continues."""

import argparse
import math
from PIL import Image, ImageDraw
from common import *
from object_model import ObjectGaussians


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument(
    "--scene", action="store_true", help="Also render the unchanged retained background"
)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cuda")
obj = ObjectGaussians(checkpoint["transform"], config()["initialization"])
obj.load_state(checkpoint["state"])
dest = args.checkpoint.parent / "checkpoint_reviews" / args.checkpoint.stem
dest.mkdir(parents=True, exist_ok=True)
anchor = json.loads((OUT / "avp/anchor.json").read_text())
target = editing_region()
center = np.asarray(target["center"])
up = np.asarray(target["up"])
reference = Camera.from_record(anchor["camera"])
offset = reference.center - center
radius = np.linalg.norm(offset)
horizontal = normalized(offset - up * np.dot(offset, up))
right = normalized(np.cross(up, horizontal))
elevation = math.radians(anchor["selected"]["elevation"])
sheet = Image.new("RGB", (4 * 256, 3 * 280), "white")
draw = ImageDraw.Draw(sheet)
scene_sheet = Image.new("RGB", sheet.size, "white") if args.scene else None
scene_draw = ImageDraw.Draw(scene_sheet) if args.scene else None
cameras = []
with torch.no_grad():
    data = obj.data()
    if args.scene:
        background = GaussianData.load(
            ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply",
            selection=np.load(OUT / "localization/keep_indices.npy"),
        )
        scene = concatenate(background, data)
    for i in range(12):
        angle = 2 * math.pi * i / 12
        position = center + radius * (
            math.cos(elevation)
            * (math.cos(angle) * horizontal + math.sin(angle) * right)
            + math.sin(elevation) * up
        )
        cam = look_at(position, center, up, 512, 49.1)
        cameras.append(cam.record())
        path = dest / f"{i:02d}.png"
        save_image(
            path,
            render(cam, data, background=torch.ones(3, device="cuda"), aux=False)[
                "rgb"
            ],
        )
        x, y = (i % 4) * 256, (i // 4) * 280
        sheet.paste(Image.open(path).resize((256, 256)), (x, y + 24))
        draw.text(
            (x + 8, y + 5), f"Step {checkpoint['step']} | az {i * 30} deg", fill="black"
        )
        if args.scene:
            scene_path = dest / f"scene_{i:02d}.png"
            save_image(scene_path, render(cam, scene, aux=False)["rgb"])
            scene_sheet.paste(Image.open(scene_path).resize((256, 256)), (x, y + 24))
            scene_draw.text(
                (x + 8, y + 5),
                f"Step {checkpoint['step']} | az {i * 30} deg",
                fill="black",
            )
    visible = obj.opacity_logit.sigmoid().squeeze(-1) > 0.1
    xyz = obj.xyz[visible].cpu().numpy()
    covariance = np.cov(xyz.T) if len(xyz) > 3 else np.zeros((3, 3))
    metadata = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "points": len(obj.xyz),
        "points_opacity_above_01": int(visible.sum()),
        "visible_local_position_covariance_eigenvalues": np.linalg.eigvalsh(
            covariance
        ).tolist(),
        "visible_local_xyz_quantiles_01_99": (
            np.quantile(xyz, [0.01, 0.99], axis=0).tolist() if len(xyz) else []
        ),
        "cameras": cameras,
        "limits": "Position spread does not prove surface quality; inspect all rendered views.",
    }
sheet.save(dest / "contact_sheet.jpg", quality=94)
if args.scene:
    scene_sheet.save(dest / "contact_sheet_scene.jpg", quality=94)
save_json(dest / "metadata.json", metadata)
print(dest, flush=True)
