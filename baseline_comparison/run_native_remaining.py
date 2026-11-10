"""Run four remaining paired cases after both case00 results were inspected."""
import concurrent.futures,json,subprocess,threading,time
from pathlib import Path

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
lock=threading.Lock();state={}


def update(key,**values):
    with lock:
        state.setdefault(key,{}).update(values)
        dest=BASE/'records/native_dge_gsle_batch.json';tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(updated_at=time.time(),jobs=state),indent=2));tmp.replace(dest)


def queue(case,gpu):
    for method in ['dge','gsle']:
        key=method+'/'+case;out=BASE/(method+'_native')/case
        subprocess.run(['python3',str(BASE/('native_'+method)/'prepare.py'),case],cwd=ROOT,check=True)
        update(key,status='waiting_for_gpu_memory',gpu=gpu)
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=26000:break
            time.sleep(20)
        log=out/'logs/batch_supervisor.log';assert not log.exists(),'Preserve prior attempt'
        with log.open('w') as f:
            proc=subprocess.Popen(['python3',str(BASE/('native_'+method)/'run.py'),case,'--gpu',gpu],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            update(key,status='running',started_at=time.time(),pid=proc.pid)
            rc=proc.wait()
        if rc:
            update(key,status='failed',exit_code=rc,finished_at=time.time());continue
        with (out/'logs/verification.log').open('w') as f:
            rc=subprocess.call([str(ROOT/'envs/gsle/bin/python'),str(BASE/'verify_native.py'),method,case],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        update(key,status='complete' if rc==0 else 'verification_failed',verification_exit_code=rc,finished_at=time.time())


if __name__=='__main__':
    for method in ['dge','gsle']:
        out=BASE/(method+'_native')/'case00'
        assert json.loads((out/'records/verification.json').read_text())['status']=='passed'
        assert (out/'records/visual_review.json').exists(),'Inspect actual case00 images before the remaining cases'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(queue,c,str(i)) for i,c in enumerate(['case02','case04','case07','case10'])]
        for f in futures:f.result()
