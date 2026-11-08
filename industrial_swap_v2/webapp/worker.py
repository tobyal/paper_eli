"""Single durable server worker; GPU subprocesses run outside HTTP requests."""
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import yaml
from . import store as s
from . import service as v


def fail(identifier, error):
    with s.transaction() as conn:
        s.sql(conn,"UPDATE stage_runs SET status='failed',error=:error,finished_at=:t WHERE id=:id",error=str(error)[-6000:],t=s.now(),id=identifier)
        row=v.run(identifier)
        api=int((Path(row['folder'])/'records/qwen_generation_once.lock').exists())
        s.sql(conn,"UPDATE generation_records SET status='failed',external_api=:api WHERE run_id=:id",api=api,id=identifier)


def elapsed(value):
    return max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(value['started_at'] or value['created_at'])).total_seconds())


def choose_gpu(cfg, value):
    stage,action=value['stage'],value['action']; inputs=s.unpack(value['inputs'])
    required=cfg[stage].get('minimum_free_gpu_gib',4) if stage in ('m3','m4') else 4
    # Reuse does not instantiate the expensive teacher/generator. CPU-only
    # imports should not wait for training-sized GPU reservations.
    if stage=='m1': required=0
    if stage=='m2' and action=='reuse' and s.asset(inputs['asset'])['kind']=='object_rgba': required=0
    if stage=='m3' and (action=='reuse' or generation_cache(value,cfg,inputs)): required=4
    if stage=='m3' and cfg['m3'].get('method')=='pbr_multiview_gaussian':
        required=max(required,cfg['m3']['pbr']['training']['minimum_free_gpu_gib'])
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    choices=[(int(free.strip())/1024,int(index)) for index,free in (line.split(',') for line in result.strip().splitlines())]
    free,gpu=max(choices)
    if free<required: return None
    return gpu


def generation_cache(value,cfg,inputs):
    if value['action']!='trellis' or s.unpack(value['parameters']).get('force_model'): return None
    image_hash=s.sha(v.stage_dir(inputs['m2'])/'object_rgba.png')
    expected={k:cfg['m3'][k] for k in ('model','revision','pipeline_type','texture_size','decimation_target')}
    for asset in s.rows("SELECT * FROM assets WHERE kind='mesh' ORDER BY created_at"):
        meta=s.unpack(asset['metadata'])
        if meta.get('input_image_sha')!=image_hash: continue
        source=meta.get('run_id')
        if not source: continue
        previous=yaml.safe_load(Path(v.run(source)['config_path']).read_text())
        if all(previous['m3'][k]==val for k,val in expected.items()) and previous['project']['seed']==cfg['project']['seed']:
            s.asset_path(asset['id'],verify=True); return asset['id']
    return None


def prepare(value, cfg):
    folder=Path(value['folder']); out=folder/'outputs'; out.mkdir(exist_ok=True)
    inputs=s.unpack(value['inputs'])
    for key,rid in inputs.items():
        if key=='asset': continue
        dependency=v.run(rid)
        if not v.verify_outputs(dependency): raise ValueError('依赖产物已变化：'+key)
        source=v.stage_dir(rid)
        if key=='materials':
            target=out/'m2'; target.mkdir(exist_ok=True)
            for name in (*v.MATERIALS,*v.SPATIAL_MATERIALS):
                if (source/name).is_file() and not (target/name).exists(): shutil.copy2(source/name,target/name)
        else:
            target=out/key
            if not target.exists(): target.symlink_to(source,target_is_directory=True)
    records=folder/'records'; records.mkdir(exist_ok=True)
    if value['stage']=='m2' and value['action']=='prepare':
        from .m2_condition_cache import reuse_conditions
        reused=reuse_conditions(cfg,out/'m2',out/'m1')
        if reused:
            with Path(value['log_path']).open('a') as log:
                log.write('Reused verified same-case AVP/RGB/depth conditions: '+reused+'; build a new prompt independently.\n')
    for name in ('trellis_model.json','trellis_dependencies.json'):
        shutil.copy2(s.BASE/'records'/name,records/name)
    aid=inputs.get('asset') or generation_cache(value,cfg,inputs)
    asset=None
    if aid:
        asset=s.asset(aid); s.asset_path(aid,verify=True); asset['metadata']=s.unpack(asset['metadata'])
    job={**{k:value[k] for k in ('id','stage','action','folder')},'parameters':s.unpack(value['parameters']),
         'asset':asset,'job_path':str(folder/'job.json')}
    (folder/'job.json').write_text(s.dump(job),encoding='utf-8')
    return job


def execute(value):
    start=time.monotonic(); folder=Path(value['folder']); cfg=yaml.safe_load(Path(value['config_path']).read_text())
    success=folder/'adapter_success.json'
    # A worker restart after successful GPU work only needs metadata finalization.
    if success.exists():
        v.finish(value['id'],folder/'outputs'/value['stage'],seconds=elapsed(value)); return True
    target=folder/'outputs'/value['stage']
    # A failed attempt can already contain partial files. Resume that attempt
    # through the adapter, never label those partial files as a cache hit.
    if not target.exists() and value['action']!='qwen' and not s.unpack(value['parameters']).get('force_model'):
        candidates=s.rows("SELECT * FROM stage_runs WHERE signature=:sig AND status='completed' ORDER BY created_at DESC",sig=value['signature'])
        for candidate in candidates:
            if v.verify_outputs(candidate):
                source=v.stage_dir(candidate['id'])
                target.parent.mkdir(exist_ok=True)
                if not target.exists(): target.symlink_to(source,target_is_directory=True)
                Path(value['log_path']).write_text('Reused verified stage cache: '+candidate['id']+'\n',encoding='utf-8')
                v.finish(value['id'],target,reused_from=candidate['id'],seconds=time.monotonic()-start); return True
    gpu=choose_gpu(cfg,value)
    if gpu is None: return False
    with s.transaction() as conn:
        s.sql(conn,"UPDATE stage_runs SET status='running',started_at=:t WHERE id=:id",t=s.now(),id=value['id'])
        s.sql(conn,"UPDATE generation_records SET status='running' WHERE run_id=:id",id=value['id'])
    job=prepare(value,cfg)
    env=os.environ.copy(); env.update(SWAP_GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',SWAP_DISABLE_QWEN='1',
        ROOT=str(s.ROOT),CUDA_HOME='/usr/local/cuda-11.8',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    env['PATH']='/usr/local/cuda-11.8/bin:'+env['PATH']
    command=[str(s.ROOT/cfg['project']['gsle_python']),'-m','industrial_swap_v2.webapp.adapter',
             '--config',value['config_path'],'--job',job['job_path']]
    with Path(value['log_path']).open('a',encoding='utf-8') as log:
        log.write(s.dump({'command':command,'gpu':gpu,'attempt':value['attempt'],'time':s.now()})+'\n'); log.flush()
        process=subprocess.Popen(command,cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with s.transaction() as conn: s.sql(conn,'UPDATE stage_runs SET pid=:pid WHERE id=:id',pid=process.pid,id=value['id'])
        code=process.wait()
    if code!=0 or not success.is_file(): raise RuntimeError(f'执行器退出码 {code}；请查看运行日志，可从已保存检查点恢复')
    v.finish(value['id'],folder/'outputs'/value['stage'],seconds=time.monotonic()-start)
    return True


def recover():
    for value in s.rows("SELECT * FROM stage_runs WHERE status='running'"):
        pid=value['pid']; command=Path(f'/proc/{pid}/cmdline') if pid else None
        if command and command.exists() and b'industrial_swap_v2.webapp.adapter' in command.read_bytes():
            # Do not start a duplicate while an orphaned, legitimate GPU subprocess is active.
            return False
        if (Path(value['folder'])/'adapter_success.json').is_file():
            try: v.finish(value['id'],Path(value['folder'])/'outputs'/value['stage'],seconds=elapsed(value))
            except Exception as exc: fail(value['id'],exc)
        else: fail(value['id'],'Worker interrupted; saved intermediates retained. Resume explicitly.')
    return True


def main():
    s.init(); lock=(s.DATA/'service/worker.lock').open('w')
    from . import m1_cases, m1_worker
    m1_cases.init()
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return
    while True:
        if not recover() or not m1_worker.recover(): time.sleep(3); continue
        if m1_worker.tick(): time.sleep(2); continue
        value=s.one("SELECT * FROM stage_runs WHERE status='pending' ORDER BY created_at LIMIT 1")
        if value:
            try:
                if not execute(value): time.sleep(5)
            except Exception as exc:
                with Path(value['log_path']).open('a',encoding='utf-8') as log: traceback.print_exc(file=log)
                fail(value['id'],exc)
        else: time.sleep(2)


if __name__=='__main__': main()
