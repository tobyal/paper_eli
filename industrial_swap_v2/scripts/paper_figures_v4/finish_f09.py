"""Finalize only after the recovered CNC M3 has actually completed."""
import os,sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import read_json,write_json
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F09_text_control'
SCRIPTS=ROOT/'industrial_swap_v2/scripts/paper_figures_v4'

def main():
    ready=OUT/'variants/C_cnc/m3_elapsed.json';deadline=time.monotonic()+1800
    while not ready.exists():
        if time.monotonic()>deadline:raise RuntimeError('CNC not complete; no completed figure is claimed')
        time.sleep(20)
    status=read_json(OUT/'generation_status.json');history=OUT/'generation_status_before_export_recovery.json'
    if not history.exists():shutil.copy2(OUT/'generation_status.json',history)
    for key in ['B_scara','C_cnc','D_conveyor']:
        assert (OUT/'variants'/key/'records/m3_status.json').exists()
        assert read_json(OUT/'variants'/key/'records/m3_status.json')['status']=='complete'
        status[key].update(status='m3_complete',exit_code=0)
    status['C_cnc']['resource_adaptation']='250000-face official export from original native checkpoint; million-face UV attempt retained'
    write_json(OUT/'generation_status.json',status)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',SWAP_GPU='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    for script in ['place_f09.py','figure_f09.py','package.py']:
        print('RUN',script,flush=True);subprocess.run([sys.executable,str(SCRIPTS/script)],cwd=ROOT,env=env,check=True)
    print('ALL PAPER FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
