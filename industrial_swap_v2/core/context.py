"""All writable paths are confined to the new project directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Context:
    def __init__(self, path):
        self.config_path = Path(path).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if self.cfg['m3']['model'] != 'microsoft/TRELLIS.2-4B':
            raise ValueError('V2 fixes the official microsoft/TRELLIS.2-4B model')
        if self.cfg['m4']['method'] != 'di_sds':
            raise ValueError('V2 refinement implements DI-SDS')
        if not self.cfg['m1']['reuse_existing_localization']:
            raise NotImplementedError('M1 adapter reuses existing LangSplat results; run the original localization first for a new target')
        for key in ('use_avp','use_scene_depth','use_background_depth','use_sam'):
            if not self.cfg['m2'][key]:
                raise NotImplementedError(f'The current M2 adapter requires {key}; implement a replacement adapter before disabling it')
        self.root = Path(os.environ.get("SWAP_PROJECT_ROOT", self.cfg["project"]["root"])).resolve()
        self.base = self.root / "industrial_swap_v2"
        self.out = self.path(self.cfg["project"]["output_dir"])
        self.records = self.path(self.cfg['project'].get('records_dir', 'records'))
        self.logs = self.path(self.cfg['project'].get('logs_dir', 'logs'))
        for folder in (self.out, self.records, self.logs):
            folder.mkdir(parents=True, exist_ok=True)

    def path(self, relative):
        path = (self.base / relative).resolve()
        try:
            path.relative_to(self.base.resolve())
        except ValueError:
            raise ValueError("Writable paths must remain inside industrial_swap_v2") from None
        return path

    def source(self, relative):
        # Preserve virtualenv interpreter symlinks: resolving bin/python to its
        # base interpreter silently bypasses that environment's site-packages.
        return (self.root / relative).absolute()

    def stage(self, name):
        path = self.out / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def env(self, gpu=None):
        env = os.environ.copy()
        env.update(ROOT=str(self.root), GSLE_OUTPUT=str(self.out / "gsle_adapter"),
                   GSLE_RECORDS=str(self.records / "gsle_adapter"),
                   CUDA_VISIBLE_DEVICES=str(gpu if gpu is not None else env.get("SWAP_GPU", self.cfg["project"]["gpu"])),
                   OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", PYTHONUNBUFFERED="1")
        return env

    def receipt(self, stage, status, **extra):
        value = {"stage": stage, "status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        write_json(self.records / f"{stage}_status.json", value)
        return value

    def complete(self, stage, files, **extra):
        artifacts = []
        for path in files:
            path = Path(path)
            assert path.is_file(), path
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        return self.receipt(stage, "complete", artifacts=artifacts, config_signature=self.signature(stage), **extra)

    def signature(self, stage):
        relevant = {'stage': self.cfg[stage], 'seed': self.cfg['project']['seed'],
                    'source_sha256': self.cfg['project']['source_sha256']}
        # Credentials and resource placement are not experiment parameters.
        if stage == 'm2':
            relevant['stage'] = {k:v for k,v in self.cfg[stage].items() if k != 'api_key_file'}
        return hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def gsle(ctx):
    os.environ.update({k: v for k, v in ctx.env().items() if k in ("ROOT", "GSLE_OUTPUT", "GSLE_RECORDS")})
    sys.path.insert(0, str(ctx.root / "gsle"))
    import common
    return common
