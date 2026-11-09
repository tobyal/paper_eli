"""Run native GE stages independently; never overwrite an existing stage log."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE = ROOT / 'baseline_comparison'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--gpu', required=True)
    p.add_argument('--through', choices=['localize', 'delete', 'add', 'export'], default='export')
    a = p.parse_args()
    out = BASE / 'gaussianeditor_native' / a.case
    assert (out / 'config.json').exists(), 'Run prepare.py first'
    status = out / 'records/status.json'
    record = json.loads(status.read_text()) if status.exists() else dict(case=a.case, stages={}, started_at=time.time())
    env = os.environ.copy()
    for key in ['TARGET_GEOMETRY', 'INSTANCE_SELECTION_LOG', 'GE_TARGET_CAMERA_DEPTH', 'GE_REUSE_ADD_ASSETS']:
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, HF_HOME=str(ROOT / 'models/huggingface'),
               TORCH_HOME=str(ROOT / 'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               U2NET_HOME=str(ROOT / 'models/rembg'), PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID=a.gpu,
               GE_ADD_CACHE=str(out / 'add'), GE_NATIVE_TRACE_DIR=str(out / 'localization'))
    sequence = ['localize', 'delete', 'add', 'export']
    for stage in sequence[:sequence.index(a.through) + 1]:
        if record['stages'].get(stage, {}).get('status') == 'complete':
            continue
        free = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu, '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 22000, 'Insufficient shared GPU memory; no process will be stopped'
        argv = [str(BASE / 'envs/gaussianeditor_add/bin/python'),
                str(BASE / 'native_gaussianeditor/stage.py'), a.case, stage]
        log = out / 'logs' / (stage + '.log')
        assert not log.exists(), 'Existing attempt retained; inspect/archive it explicitly before retry'
        record.update(status='running', active_stage=stage, gpu=a.gpu)
        entry = dict(status='running', started_at=time.time(), argv=argv)
        record['stages'][stage] = entry
        status.write_text(json.dumps(record, indent=2))
        with log.open('w') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            status.write_text(json.dumps(record, indent=2))
            rc = child.wait()
        entry.update(status='complete' if rc == 0 else 'failed', seconds=time.time() - entry['started_at'], exit_code=rc)
        record['status'] = 'running' if rc == 0 else 'failed'
        status.write_text(json.dumps(record, indent=2))
        with (out / 'run.log').open('a') as stream:
            stream.write('\n=== ' + stage + ' ===\n' + log.read_text())
        if rc:
            raise SystemExit(rc)
    record.update(status='complete' if a.through == 'export' else 'prepared_through_' + a.through,
                  seconds=time.time() - record['started_at'])
    status.write_text(json.dumps(record, indent=2))
    print(record['status'], out, flush=True)


if __name__ == '__main__':
    main()
