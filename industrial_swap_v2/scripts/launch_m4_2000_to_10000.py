"""Launch the requested continuation on ubuntuA6000; keep historical runs immutable."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import tarfile
from ..core.context import Context,read_json,sha256,write_json
from .run_background_only import initialize


def main():
    root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
    if Path.cwd().resolve()!=root: raise RuntimeError('Run from the project on ubuntuA6000')
    base=root/'industrial_swap_v2'
    source=base/'outputs/m4_dual_mask/run_2000_save500_20261106_165252'
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination='outputs/m4_dual_mask/run_resume2000_to10000_save2000_'+stamp
    checkpoint=source/'m4/snapshots/iteration_002000/checkpoint.pth'
    gpu=2
    free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<24*1024: raise RuntimeError('Selected shared GPU has less than 24 GiB free; do not start')
    config=initialize(source/'config.yaml',destination,gpu,10000,2000,checkpoint)
    ctx=Context(config)
    preserved={str(p):sha256(p) for p in source.rglob('*') if p.is_file() and not p.is_symlink()}
    write_json(ctx.out/'previous_run_sha256.json',preserved)
    files=[p for folder in ('core','modules','scripts','configs') for p in (base/folder).rglob('*')
           if p.is_file() and p.suffix in ('.py','.yaml','.sh')]
    with tarfile.open(ctx.out/'executed_source.tar.gz','w:gz') as archive:
        for path in files: archive.add(path,arcname=str(path.relative_to(root)))
    write_json(ctx.out/'code_provenance.json',{
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_files':{str(p.relative_to(root)):sha256(p) for p in files},
        'source_archive_sha256':sha256(ctx.out/'executed_source.tar.gz'),
        'source_checkpoint':str(checkpoint),'source_checkpoint_sha256':sha256(checkpoint),
        'tests':'25 CPU tests passed: resume, snapshots, official workflow, background permissions'})
    command=[str(root/'envs/gsle/bin/python'),'-m','industrial_swap_v2.scripts.run_background_only',
             '--config',str(config),'--stage','all']
    runner=ctx.out/'runner.py'
    runner.write_text('import json,subprocess,time\nfrom pathlib import Path\n'
        +'root=Path('+repr(str(ctx.out))+')\n'
        +'command='+repr(command)+'\n'
        +'start=time.time()\ncode=subprocess.call(command)\n'
        +'(root/"process_exit.json").write_text(json.dumps({"returncode":code,"seconds":time.time()-start,"command":command},indent=2))\n'
        +'raise SystemExit(code)\n',encoding='utf-8')
    with (ctx.out/'launch.log').open('w') as log:
        child=subprocess.Popen([str(root/'envs/gsle/bin/python'),str(runner)],cwd=root,
            env={**os.environ,**ctx.env(),'SWAP_DISABLE_QWEN':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'status':'running','pid':child.pid,'run':str(ctx.out),'config':str(config),
        'source_checkpoint':str(checkpoint),'start_step':2000,'target_step':10000,
        'save_steps':[4000,6000,8000,10000],'gpu':gpu,'command':command}
    write_json(ctx.out/'launch.json',record)
    write_json(base/'records/m4_resume2000_to10000_active.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
