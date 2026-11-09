"""Package completed GSLE artifacts without redistributing pretrained weights."""

import hashlib, json, os, subprocess, sys, tarfile, time
from pathlib import Path

root = Path(os.environ["ROOT"])
out = root / "outputs/gsle"
required = [
    "localization/target.json",
    "avp/anchor.json",
    "inpainting/edited_anchor.png",
    "foreground/object_rgba.png",
    "coarse/final.pth",
    "composition/scene.ply",
    "refine/final.pth",
    "final/scene.ply",
    "final/review/summary.json",
    "final/renders/comparison_original_speed_2fps.mp4",
    "final/renders/object_360.mp4",
]
for name in required:
    assert (out / name).is_file(), f"Missing actual artifact: {name}"
summary = json.loads((out / "final/review/summary.json").read_text())
assert summary["stages"]["coarse"]["optimizer_updates"] == 6500
assert summary["stages"]["refine"]["optimizer_updates"] == 2000
assert summary["composition"]["retained_background_bitwise_identical"]
assert (root / "records/gsle/final_artifact_verification.json").is_file()
delivery = json.loads((root / "records/gsle/deliverables.json").read_text())
assert (root / "records/gsle/final_visual_review.json").is_file()
if (out / "ablations/text_only_cfg").exists():
    alternative = json.loads(
        (out / "ablations/text_only_cfg/final/review/summary.json").read_text()
    )
    assert alternative["stages"]["refine"]["optimizer_updates"] == 2000
    assert (
        root / "records/gsle/ablations/text_only_cfg/final_artifact_verification.json"
    ).is_file()
    assert (out / "ablation_comparison/comparison.json").is_file()
subprocess.run([sys.executable, str(root / "gsle/status.py")], check=True)
manifest_path = root / "records/gsle/package_manifest.json"
files = []
for folder in ("gsle", "records/gsle", "logs/gsle"):
    files.extend(
        path
        for path in (root / folder).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pstats", ".completed")
    )
for folder in (
    "localization_proposals",
    "localization",
    "avp",
    "inpainting",
    "foreground",
    "coarse",
    "composition",
    "refine",
    "final",
    "attempts",
    "localization_attempts",
    "ablations",
    "ablation_comparison",
):
    for path in (out / folder).rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        if path.name.startswith("checkpoint_") and path.name != "checkpoint_003500.pth":
            continue
        files.append(path)
files.extend([root / "records/camera_split.json", root / "scripts/common.sh"])
for name in ("cameras.bin", "images.bin", "points3D.bin"):
    files.append(root / "data/industrial/sparse/0" / name)
files = sorted(set(files) - {manifest_path})
assert all(path.is_file() or path.is_symlink() for path in files)
manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "file_count_including_manifest": len(files) + 1,
    "uncompressed_bytes_excluding_manifest": sum(
        path.stat().st_size for path in files if not path.is_symlink()
    ),
    "files": [str(path.relative_to(root)) for path in files + [manifest_path]],
    "server_root": str(root),
    "selected_variant": delivery["selected_variant"],
    "visual_quality_review": delivery["visual_review"],
    "symlink_policy": "Preserve shared inputs using relative links within the project, without duplicating their contents.",
    "excluded": [
        "pretrained model cache",
        "input video and full image dataset",
        "LangSplat training checkpoints/features (retained on server)",
        "non-final object checkpoints except coarse3500",
        "diagnostic-only optimizer probes",
    ],
    "rerun_instructions": "gsle/README.txt",
    "status": "actual coarse6500 and DI-SDS2000 artifacts present; retained background exact",
}
manifest_path.write_text(json.dumps(manifest, indent=2))
files.append(manifest_path)
archive = root / "gsle_results.tar.gz"
temporary = archive.with_suffix(".gz.tmp")
with tarfile.open(temporary, "w:gz", compresslevel=4) as tar:
    for index, path in enumerate(files):
        if path.is_symlink():
            assert path.resolve().is_relative_to(root.resolve()), path
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.linkname = os.path.relpath(path.resolve(), path.parent)
            tar.addfile(info)
        else:
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        if index % 100 == 0:
            print("Packaged", index, "/", len(files), flush=True)
temporary.replace(archive)
digest = hashlib.sha256()
with archive.open("rb") as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
checksum = digest.hexdigest()
(root / "gsle_results.tar.gz.sha256").write_text(f"{checksum}  {archive.name}\n")
print(
    json.dumps(
        {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum}
    ),
    flush=True,
)
