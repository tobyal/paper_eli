"""M1 jobs share the existing single GPU worker and restart recovery."""
import os
from pathlib import Path
import subprocess
import sys
from . import store as s


def finish(row):
    success=(Path(row['folder'])/'success.json').is_file()
    with s.transaction() as conn:
        s.sql(conn,'UPDATE m1_jobs SET status=:status,error=:error,finished_at=:time WHERE id=:id',
              status='completed' if success else 'failed',error=None if success else 'M1 执行未完成，请查看日志。可切换描述或 Mask 后重试。',time=s.now(),id=row['id'])


def recover():
    for row in s.rows("SELECT * FROM m1_jobs WHERE status='running'"):
        proc=Path(f"/proc/{row['pid']}/cmdline")
        if proc.exists() and b'industrial_swap_v2.webapp.m1_job' in proc.read_bytes():return False
        # The coordinator might have exited while its inherited GPU child survives.
        for item in Path('/proc').glob('[0-9]*/cmdline'):
            try:command=item.read_bytes()
            except (PermissionError,FileNotFoundError,ProcessLookupError):continue
            if str(row['folder']).encode() in command and any(n in command for n in (b'industrial_swap_v2.scripts.m1_language',b'industrial_swap_v2.scripts.m1_assisted_localize')):return False
        finish(row)
    return True


def tick():
    row=s.one("SELECT * FROM m1_jobs WHERE status='pending' ORDER BY created_at LIMIT 1")
    if not row:return False
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free,gpu=max((int(free),int(index)) for index,free in (line.split(',') for line in output.strip().splitlines()))
    if free<8*1024:return True
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),SWAP_GPU=str(gpu),SWAP_DISABLE_QWEN='1',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_HOME='/usr/local/cuda-11.8')
    env['PATH']='/usr/local/cuda-11.8/bin:'+env['PATH'];folder=Path(row['folder'])
    command=[sys.executable,'-m','industrial_swap_v2.webapp.m1_job','--job',str(folder/'job.json')]
    with s.transaction() as conn:
        claimed=s.sql(conn,"UPDATE m1_jobs SET status='running',started_at=:time WHERE id=:id AND status='pending'",time=s.now(),id=row['id'])
        if claimed.rowcount!=1:return True
    with (folder/'run.log').open('a',encoding='utf-8') as log:
        log.write(s.dump({'command':command,'gpu':gpu,'stages':['m1'],'started_at':s.now()})+'\n');log.flush()
        try:
            process=subprocess.Popen(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with s.transaction() as conn:s.sql(conn,'UPDATE m1_jobs SET pid=:pid WHERE id=:id',pid=process.pid,id=row['id'])
            process.wait()
        except OSError as error:
            log.write('M1 launch failed: '+str(error)+'\n')
    finish(row);return True
