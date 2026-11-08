"""Run explicitly selected pending native stages on one GPU; preserve Web receipts.

The idle batch-only postprocessor is paused and resumed in finally. The primary
Web worker remains running. No stage is marked completed before adapter success.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import yaml
from industrial_swap_v2.webapp import store as s, service as v, worker as w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--keys', nargs='+', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--idle-postprocess-pid', type=int, required=True)
    a = p.parse_args(); batch = a.batch.resolve()
    process = Path('/proc') / str(a.idle_postprocess_pid)
    command = (process / 'cmdline').read_bytes()
    assert process.stat().st_uid == os.getuid()
    assert b'industrial_swap_v2.scripts.postprocess_qwen_batch\x00' in command
    states = json.loads((batch / 'postprocess.json').read_text())['items']
    assert all(x['status'] in ('waiting_m3', 'awaiting_pose_review', 'completed') for x in states.values())
    os.kill(a.idle_postprocess_pid, signal.SIGSTOP)
    try:
        for key in a.keys:
            for stage in ('m2', 'm3'):
                while True:
                    item = next(x for x in json.loads((batch / 'manifest.json').read_text())['items'] if x['key'] == key)
                    rid = item['runs'].get(stage)
                    if not rid:
                        time.sleep(5); continue  # existing watch-m3 creates its normal stage
                    value = v.run(rid)
                    if value['status'] == 'completed': break
                    if value['status'] == 'failed': raise RuntimeError(key + '/' + stage + ' already failed')
                    if value['status'] != 'pending': time.sleep(5); continue
                    cfg = yaml.safe_load(Path(value['config_path']).read_text())
                    free = float(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free',
                                                          '--format=csv,noheader,nounits'], text=True)) / 1024
                    if stage == 'm3' and free < max(24, cfg['m3']['minimum_free_gpu_gib']):
                        time.sleep(10); continue
                    with s.transaction() as conn:
                        updated = s.sql(conn, "UPDATE stage_runs SET status='importing',started_at=:t WHERE id=:r AND status='pending'",
                                        t=s.now(), r=rid)
                        if updated.rowcount != 1: continue
                        s.sql(conn, "UPDATE generation_records SET status='importing' WHERE run_id=:r", r=rid)
                    started = time.monotonic()
                    try:
                        job = w.prepare(value, cfg)
                        env = os.environ.copy()
                        env.update(SWAP_GPU=str(a.gpu), CUDA_VISIBLE_DEVICES=str(a.gpu), SWAP_DISABLE_QWEN='1',
                                   ROOT=str(s.ROOT), CUDA_HOME='/usr/local/cuda-11.8', OMP_NUM_THREADS='8',
                                   OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1')
                        env['PATH'] = '/usr/local/cuda-11.8/bin:' + env['PATH']
                        cmd = [str(s.ROOT / cfg['project']['gsle_python']), '-m', 'industrial_swap_v2.webapp.adapter',
                               '--config', value['config_path'], '--job', job['job_path']]
                        with Path(value['log_path']).open('a') as log:
                            log.write(json.dumps({'batch_parallel_worker': True, 'command': cmd, 'gpu': a.gpu, 'at': s.now()}) + '\n')
                            log.flush()
                            subprocess.run(cmd, cwd=s.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        assert (Path(value['folder']) / 'adapter_success.json').is_file()
                        v.finish(rid, Path(value['folder']) / 'outputs' / stage, seconds=time.monotonic() - started)
                        print(key, stage, 'completed', flush=True)
                    except Exception as exc:
                        with Path(value['log_path']).open('a') as log: traceback.print_exc(file=log)
                        w.fail(rid, exc)
                        raise
                    break
    finally:
        os.kill(a.idle_postprocess_pid, signal.SIGCONT)
        print('Batch postprocessor resumed', flush=True)


if __name__ == '__main__': main()
