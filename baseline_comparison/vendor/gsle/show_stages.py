"""Arrange existing experiment images for review; no model inference."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

root = Path(__file__).resolve().parents[1]
dest = root / "results/gsle/stages"
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc", size=12)


def panel(output, images, width=14, height=4):
    figure, axes = plt.subplots(1, len(images), figsize=(width, height), squeeze=False)
    for ax, (path, title) in zip(axes[0], images):
        ax.imshow(Image.open(dest / path).convert("RGB"))
        ax.set_title(title, fontproperties=font, pad=9)
        ax.axis("off")
    figure.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.01, wspace=0.015)
    figure.savefig(dest / output, dpi=110, facecolor="white")
    plt.close(figure)


panel(
    "01_localization.png",
    [
        ("01_localization/frame_000151_original.png", "原始场景"),
        ("01_localization/frame_000151_localized.png", "三维定位：红色高亮旧设备"),
        ("01_localization/frame_000151_removed.png", "移除旧设备后的场景"),
    ],
    width=15,
    height=3.4,
)
panel(
    "02_avp.png",
    [
        ("02_avp/anchor.png", "自动选定的 Anchor View"),
        ("02_avp/bbox_mask.png", "投影编辑区域：白色为待编辑区"),
        ("02_avp/depth_control.png", "归一化反转深度：ControlNet 输入"),
    ],
    width=12,
    height=4.3,
)
panel(
    "03_inpainting.png",
    [
        ("02_avp/anchor.png", "生成前的参考视角"),
        ("03_inpainting/edited_anchor.png", "深度条件 Inpainting 输出，种子 42"),
    ],
    width=9,
    height=4.7,
)
panel(
    "04_sam.png",
    [
        ("04_sam/reference_white.png", "SAM 提取的新物体：白底展示"),
        ("04_sam/mask.png", "SAM 前景 Mask"),
    ],
    width=9,
    height=4.7,
)

server = "/mnt/sda/ubuntu/lkx/papers/paper_eli/outputs/gsle"
index = [
    (
        "3D Localization",
        "localization",
        [
            "target.json",
            "edit_region.json",
            "target_indices.npy",
            "keep_indices.npy",
            "target.ply",
            "scene_without_target.ply",
            "multiview/contact_sheet.jpg",
        ],
    ),
    (
        "AVP",
        "avp",
        [
            "anchor.json",
            "anchor.png",
            "bbox_mask.png",
            "depth.npz",
            "depth_control.png",
            "candidates.json",
        ],
    ),
    (
        "Depth Inpainting",
        "inpainting",
        ["edited_anchor.png", "generation.json", "candidate_sheet.jpg", "candidates/"],
    ),
    (
        "SAM",
        "foreground",
        [
            "object_rgba.png",
            "reference_white.png",
            "mask.png",
            "automatic_prompts.json",
            "segmentation.json",
        ],
    ),
    (
        "Coarse 3D-SDS",
        "coarse",
        [
            "final.pth",
            "final_object_local.ply",
            "final_object_world.ply",
            "checkpoint_003500.pth",
            "training.jsonl",
        ],
    ),
    (
        "Scene Composition",
        "composition",
        ["scene.ply", "object_world.ply", "composition.json", "renders/"],
    ),
    (
        "DI-SDS selected text_only",
        "ablations/text_only_cfg",
        [
            "refine/final.pth",
            "refine/final_object_local.ply",
            "refine/training.jsonl",
            "final/scene.ply",
            "final/object_world.ply",
            "final/review/",
            "final/renders/",
        ],
    ),
    (
        "DI-SDS failed drop_all_eq7",
        "refine",
        ["final.pth", "final_object_local.ply", "training.jsonl"],
    ),
]
rows = [
    "GSLE 实际中间结果索引",
    "",
    "本文件记录实际文件位置；本地 stages 目录主要保存原始预览和元数据，完整三维模型/训练日志在服务器。",
    "展示图只排列真实阶段输出，未进行新的模型推理。",
    "",
]
for label, folder, files in index:
    rows.extend([label] + [f"  {server}/{folder}/{file}" for file in files] + [""])
rows += ["LangSplat 场景语言场："] + [
    f"  {server}/langsplat_{i}/chkpnt30000.pth" for i in (1, 2, 3)
]
rows += [
    "",
    "本机最终交付：",
    f'  {root / "results/gsle"}',
    "",
    "完整结果归档：",
    "/mnt/sda/ubuntu/lkx/papers/paper_eli/gsle_results.tar.gz",
]
(dest / "INDEX.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
(dest / "display_manifest.json").write_text(
    json.dumps(
        {
            "source": "actual experiment outputs",
            "panels": "01-04 only arrange and rescale original images with labels",
            "complete_paths": index,
            "server_root": server,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("Saved 4 stage comparison panels and INDEX.txt to", dest)
