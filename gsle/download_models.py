"""Download the immutable GSLE manifest, verifying large files with SHA256."""

import fnmatch, json, os, subprocess, time
from pathlib import Path
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

root = Path(os.environ["ROOT"])
pins = json.loads((root / "records/gsle/model_pins.json").read_text())
record_path = root / "records/gsle/models.json"
records = json.loads(record_path.read_text()) if record_path.exists() else {}
for name, pin in pins.items():
    repo, rev = pin["repository"], pin["revision"]
    entries = [
        f
        for f in pin["available_files"]
        if any(fnmatch.fnmatch(f["name"], p) for p in pin["patterns"])
    ]
    for pattern in pin["patterns"]:
        assert any(fnmatch.fnmatch(f["name"], pattern) for f in entries), (
            name,
            "missing pattern",
            pattern,
        )
    cache = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--"))
    files = []
    for entry in entries:
        filename, size = entry["name"], entry["bytes"]
        url = hf_hub_url(repo, filename, revision=rev)
        if size and size > 50 * 1024 * 1024:
            metadata = get_hf_file_metadata(url)
            assert len(metadata.etag) == 64
            blob = cache / "blobs" / metadata.etag
            blob.parent.mkdir(parents=True, exist_ok=True)
            if not blob.exists():
                temporary = blob.with_name(blob.name + ".incomplete")
                print(name, filename, size, flush=True)
                command = [
                    "aria2c",
                    "--continue=true",
                    "--auto-file-renaming=false",
                    "--max-connection-per-server=8",
                    "--split=8",
                    "--min-split-size=32M",
                    "--file-allocation=none",
                    "--summary-interval=60",
                    "--console-log-level=warn",
                    "--max-tries=10",
                    "--retry-wait=3",
                    "--all-proxy=" + os.environ["https_proxy"],
                    "--checksum=sha-256=" + metadata.etag,
                    "--dir=" + str(blob.parent),
                    "--out=" + temporary.name,
                ]
                for attempt in range(5):
                    result = subprocess.run(
                        command
                        + [url + "?download=true&retry=" + str(int(time.time()))]
                    )
                    if result.returncode == 0:
                        break
                    print("Retrying transport", name, filename, attempt + 1, flush=True)
                    time.sleep(3)
                result.check_returncode()
                temporary.replace(blob)
            assert blob.stat().st_size == size
            link = cache / "snapshots" / rev / filename
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(os.path.relpath(blob, link.parent))
            files.append({"name": filename, "bytes": size, "sha256": metadata.etag})
        else:
            hf_hub_download(repo, filename, revision=rev)
            files.append(entry)
    records[name] = {
        "repository": repo,
        "revision": rev,
        "path": str(cache / "snapshots" / rev),
        "files": files,
    }
    record_path.write_text(json.dumps(records, indent=2))
    print("COMPLETED", name, flush=True)
