"""Select visible calibrated views automatically and inspect target removal."""

from PIL import Image, ImageDraw
from common import *

dest = OUT / "localization/multiview"
dest.mkdir(exist_ok=True)
target = json.loads((OUT / "localization/target.json").read_text())
center = np.asarray(target["center"])
corners = np.asarray(target["bbox_corners"])
source = GaussianData.load(
    ROOT / "outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply"
)
ids = np.load(OUT / "localization/target_indices.npy")
keep = np.load(OUT / "localization/keep_indices.npy")
obj = source.subset(ids)
retained = source.subset(keep)
flags = torch.zeros(len(source.xyz), 3, device="cuda")
flags[torch.as_tensor(ids, device="cuda"), 0] = 1
ones = torch.ones(len(obj.xyz), 3, device="cuda")
candidates = []
with torch.no_grad():
    for camera in source_cameras(468):
        xy, z = camera.project(corners)
        if (
            (z <= 0.01).any()
            or (xy[:, 0] < 0).any()
            or (xy[:, 0] >= camera.width).any()
            or (xy[:, 1] < 0).any()
            or (xy[:, 1] >= camera.height).any()
        ):
            continue
        if not 0.005 < float(bbox_mask(camera, corners).mean()) < 0.5:
            continue
        isolated = raster(camera, obj, ones)[0]
        visible = raster(camera, source, flags)[0]
        ratio = float(visible.sum() / isolated.sum().clamp_min(1e-6))
        area = float((visible > 0.1).float().mean())
        candidates.append(
            {
                "camera": camera.record(),
                "visibility": ratio,
                "object_area": area,
                "direction": normalized(camera.center - center).tolist(),
                "eligible": ratio >= 0.7 and area >= 0.005,
            }
        )
eligible = [row for row in candidates if row["eligible"]]
assert eligible, "No visible original camera for localization QA"
selected = [
    next(
        (
            row
            for row in eligible
            if row["camera"]["name"] == config()["reference_image"]
        ),
        max(eligible, key=lambda row: row["object_area"]),
    )
]
while len(selected) < min(6, len(eligible)):
    remaining = [row for row in eligible if row not in selected]
    chosen = max(
        remaining,
        key=lambda row: min(
            1 - np.dot(row["direction"], old["direction"]) for old in selected
        )
        * math.sqrt(row["object_area"]),
    )
    selected.append(chosen)
save_json(
    dest / "selected_cameras.json",
    {
        "selection": "complete projected box, >=70% object visibility, then greedy angular diversity weighted by apparent area",
        "candidates": candidates,
        "selected": selected,
    },
)
sheet = Image.new("RGB", (468 * 3, 288 * len(selected)), "white")
draw = ImageDraw.Draw(sheet)
with torch.no_grad():
    for row, item in enumerate(selected):
        cam = Camera.from_record(item["camera"]).resize(938, 528)
        stem = Path(cam.name).stem
        colors = source.colors(cam.center)
        highlight = colors.clone()
        highlight[ids] = 0.4 * highlight[ids] + 0.6 * torch.tensor(
            [1.0, 0.0, 0.0], device="cuda"
        )
        images = [
            raster(cam, source, colors),
            raster(cam, source, highlight),
            render(cam, retained, aux=False)["rgb"],
        ]
        for column, (label, rgb) in enumerate(
            zip(("original", "localized", "removed"), images)
        ):
            path = dest / f"{stem}_{label}.png"
            save_image(path, rgb)
            sheet.paste(
                Image.open(path).resize((468, 264)), (column * 468, row * 288 + 24)
            )
            draw.text(
                (column * 468 + 6, row * 288 + 5), f"{stem} | {label}", fill="black"
            )
sheet.save(dest / "contact_sheet.jpg", quality=93)
print(
    json.dumps(
        {
            "selected_views": [row["camera"]["name"] for row in selected],
            "eligible_views": len(eligible),
            "contact_sheet": str(dest / "contact_sheet.jpg"),
        }
    ),
    flush=True,
)
