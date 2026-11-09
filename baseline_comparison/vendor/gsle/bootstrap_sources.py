"""Fetch immutable official dependency sources through the project proxy."""

import json, os, subprocess
from pathlib import Path

root = Path(os.environ["ROOT"])
pins = json.loads((root / "gsle/upstream.json").read_text())
records = {}
for name, pin in pins.items():
    path = root / "repos" / name
    if not (path / ".git").exists():
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "remote",
                "add",
                "origin",
                "https://github.com/" + pin["repository"] + ".git",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", pin["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == pin["commit"], (name, head)
    if name == "LangSplat":
        subprocess.run(
            ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    records[name] = {**pin, "path": str(path)}
    print(name, head, flush=True)
(root / "records/gsle/upstream.json").write_text(json.dumps(records, indent=2))
