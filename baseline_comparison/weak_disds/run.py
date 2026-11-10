"""Run a separate weak DI-SDS experiment from immutable native Coarse outputs."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(os.environ.get("ROOT", "/mnt/sda/ubuntu/lkx/papers/paper_eli"))
BASE = ROOT / "baseline_comparison"
CODE = BASE / "weak_disds"
EXP = BASE / "gsle_weak_disds"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=["case00", "case02", "case04", "case07", "case10"])
    p.add_argument("--gpu", required=True)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--label")
    p.add_argument("--resume-pilot", type=Path)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--sds-weight", type=float, default=0.0001)
    a = p.parse_args()
    assert a.iterations in (500, 1000)
    assert 0 < a.sds_weight <= 0.1
    source = BASE / "gsle_native" / a.case
    out = EXP / (a.label or a.case)
    assert out.parent == EXP and not out.exists(), f"Refusing to overwrite {out}"
    pinned = EXP / "code_native"
    upstream = BASE / "vendor/gsle_native"
    EXP.mkdir(exist_ok=True)
    if not pinned.exists():
        shutil.copytree(upstream, pinned, ignore=shutil.ignore_patterns("__pycache__"))
    for path in upstream.glob("*.py"):
        assert sha(path) == sha(pinned/path.name), "Pinned DI-SDS dependency changed"
    out.mkdir()
    for name in ("records", "logs", "refine/previews"):
        (out/name).mkdir(parents=True, exist_ok=True)
    for name in ("localization", "avp", "inpainting", "foreground", "coarse"):
        (out/name).symlink_to(source/name, target_is_directory=True)
    cfg = json.loads((source/"config.json").read_text())
    cfg["refine"].update(iterations=a.iterations, sds_weight=a.sds_weight)
    cfg["optimizer"]["color_lr"] = 0.001
    cfg["weak_disds"] = dict(source=str(source), trainable=["color_logit"],
        frozen=["xyz", "log_scale", "rotation", "opacity_logit"],
        independent_experiment=True, original_paper_baseline_unchanged=True)
    write(out/"config.json", cfg)
    paths = [source/"config.json", source/"coarse/final.pth",
        source/"coarse/final_object_world.ply", source/"refine/final.pth",
        source/"refine/final_object_world.ply"]
    for name in ("localization", "avp", "inpainting", "foreground"):
        paths += [x for x in (source/name).rglob("*") if x.is_file()]
    manifest = {str(path):sha(path) for path in paths}
    write(out/"records/source_sha256.json", manifest)
    write(out/"records/code_sha256.json", {str(x):sha(x) for d in (CODE,pinned) for x in d.glob("*.py")})
    (out/"records/training_changes.patch").write_text("".join(difflib.unified_diff(
        (pinned/"train_object.py").read_text().splitlines(True),
        (CODE/"train.py").read_text().splitlines(True),
        fromfile="native/train_object.py", tofile="weak_disds/train.py")))
    if a.resume_pilot:
        pilot = a.resume_pilot.resolve()
        assert pilot.parent == EXP.resolve()
        pcfg = json.loads((pilot/"config.json").read_text())
        assert pcfg["weak_disds"]["source"] == str(source)
        assert pcfg["refine"]["iterations"] == 500 and a.iterations == 1000
        assert pcfg["refine"]["sds_weight"] == a.sds_weight, "Resume requires the same SDS weight"
        assert (pilot/"records/gate.json").exists(), "Pilot requires reviewed acceptance gate"
        assert json.loads((pilot/"records/gate.json").read_text())["passed"]
        shutil.copy2(pilot/"refine/training.jsonl", out/"refine/training.jsonl")
        for path in (pilot/"refine/previews").glob("*.png"):
            shutil.copy2(path,out/"refine/previews"/path.name)
        shutil.copy2(pilot/"refine/checkpoint_000500.pth",out/"refine/checkpoint_000500.pth")
    env = os.environ.copy()
    env.update(ROOT=str(ROOT), CUDA_VISIBLE_DEVICES=a.gpu, GSLE_OUTPUT=str(out),
        GSLE_RECORDS=str(out/"records"), GSLE_CONFIG=str(out/"config.json"),
        HF_HOME=str(ROOT/"models/huggingface"), TORCH_HOME=str(ROOT/"models/torch"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8", PYTHONUNBUFFERED="1", PYTHONPATH=str(pinned), WANDB_MODE="disabled")
    python = ROOT/"envs/gsle/bin/python"
    train = [str(python),str(CODE/"train.py"),"refine","--microbatch",str(a.microbatch)]
    if a.resume_pilot:
        train += ["--resume",str(a.resume_pilot/"refine/final.pth")]
    commands = [("train", train), ("compose",[str(python),str(pinned/"compose.py"),"refine"]),
        ("export",[str(python),str(CODE/"export.py")])]
    status = dict(case=a.case, source=str(source), gpu=a.gpu, iterations=a.iterations,
        sds_weight=a.sds_weight,
        resume_pilot=str(a.resume_pilot) if a.resume_pilot else None, started_at=time.time(), stages={})
    status_path = out/"records/status.json"
    for name,cmd in commands:
        status.update(status="running",active_stage=name)
        status["stages"][name] = dict(argv=cmd,started_at=time.time())
        write(status_path,status)
        with (out/"logs"/(name+".log")).open("w") as log:
            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status["pid"] = proc.pid
            write(status_path,status)
            rc=proc.wait()
        status["stages"][name].update(exit_code=rc,seconds=time.time()-status["stages"][name]["started_at"])
        if rc:
            status.update(status="failed")
            write(status_path,status)
            raise SystemExit(rc)
        write(status_path,status)
    changed = [path for path,digest in manifest.items() if sha(path)!=digest]
    write(out/"records/source_preservation.json",dict(passed=not changed,checked=len(manifest),changed=changed))
    assert not changed, changed
    status.update(status="complete",active_stage=None,seconds=time.time()-status["started_at"])
    write(status_path,status)
    print(json.dumps(status),flush=True)


if __name__ == "__main__":
    main()
