"""Unified entry, stage execution in isolated interpreters with durable logs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from industrial_swap_v2.core.context import Context, read_json, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.yaml")
parser.add_argument("--stage", choices=("m1", "m2", "m3", "m4", "all"), required=True)
parser.add_argument("--prepare-only", action="store_true", help="Prepare M2 conditions without calling the API")
args = parser.parse_args()
if args.prepare_only and args.stage != 'm2':
    parser.error('--prepare-only is supported only with --stage m2')
ctx = Context(args.config)
modules = {"m1": "m1_localization", "m2": "m2_image_generation", "m3": "m3_trellis_generation", "m4": "m4_refinement"}
for stage in modules if args.stage == "all" else [args.stage]:
    receipt = ctx.records / f"{stage}_status.json"
    if receipt.exists() and read_json(receipt).get("status") == "complete":
        saved = read_json(receipt)
        if saved.get('config_signature',ctx.signature(stage)) != ctx.signature(stage):
            raise RuntimeError(f'{stage}: configuration changed since saved completion')
        if all(Path(item["path"]).is_file() and Path(item["path"]).stat().st_size == item["bytes"] and sha256(item['path'])==item['sha256'] for item in saved["artifacts"]):
            print(stage, "already complete; reusing saved artifacts", flush=True)
            continue
        raise RuntimeError(f"{stage}: saved artifacts changed; use a new output directory")
    command = [str(ctx.source(ctx.cfg["project"]["gsle_python"])), "-m", f"industrial_swap_v2.modules.{modules[stage]}", "--config", str(ctx.config_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    ctx.receipt(stage,'running',command=command,prepare_only=args.prepare_only)
    with (ctx.logs / f"{stage}.log").open("a", encoding="utf-8") as log:
        log.write("\n" + json.dumps({"command": command}) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ctx.root, env=ctx.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    if result:
        ctx.receipt(stage,'failed',exit_code=result,log=str(ctx.logs/f'{stage}.log'))
        raise SystemExit(result)
