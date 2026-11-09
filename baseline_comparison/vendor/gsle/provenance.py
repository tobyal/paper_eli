"""Capture the exact source/config/model pins used when a stage starts."""

import hashlib, json, os, socket, subprocess, time
from pathlib import Path


def capture(stage, destination=None):
    root = Path(os.environ["ROOT"])
    files = {}
    for path in sorted(Path(__file__).parent.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    record = {
        "stage": stage,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_head": git("rev-parse", "HEAD"),
        "git_worktree_status": git("status", "--short"),
        "source_sha256": files,
        "output_root": os.environ.get("GSLE_OUTPUT", str(root / "outputs/gsle")),
        "records_root": os.environ.get("GSLE_RECORDS", str(root / "records/gsle")),
    }
    for key, name in [
        ("config", os.environ.get("GSLE_CONFIG", "gsle/config.json")),
        ("upstream", "records/gsle/upstream.json"),
        ("models", "records/gsle/models.json"),
        ("camera_source", "gsle/camera_source.json"),
    ]:
        path = root / name
        if path.exists():
            record[key] = json.loads(path.read_text())
    dest = (
        Path(destination)
        if destination
        else root / f"records/gsle/provenance_{stage}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    args = parser.parse_args()
    result = capture(args.stage)
    print("Captured", len(result["source_sha256"]), "source hashes for", args.stage)
