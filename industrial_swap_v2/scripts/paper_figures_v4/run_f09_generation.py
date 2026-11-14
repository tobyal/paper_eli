"""Three requested original F09 variants; durable single Qwen POST per variant."""
import os,sys,json,time,shutil,subprocess,traceback
from pathlib import Path
import yaml
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,write_json,sha256
from industrial_swap_v2.core.m2_prompt import build_prompt
from industrial_swap_v2.core.qwen_client import generate
from industrial_swap_v2.scripts.paper_figures_v3.prepare_f09 import VARIANTS
BASE=ROOT/'industrial_swap_v2';OUT=BASE/'outputs/paper_figures_v4/F09_text_control'
PREP=BASE/'outputs/paper_figures_v3/F09_text_variants/qwen_direct_adaptive_v2/preparation'

def prepare(key,text,gpu):
    dest=OUT/'variants'/key;dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'config.yaml'
    if not cp.exists():
        cfg=yaml.safe_load((PREP/'config.yaml').read_text());rel=str(dest.relative_to(BASE))
        cfg['project'].update(output_dir=rel,records_dir=rel+'/records',logs_dir=rel+'/logs',gpu=gpu)
        cfg['m2'].update(replacement_text=text,preparation_only=False,max_api_requests=1,
            api_key_file='/mnt/sda/ubuntu/lkx/papers/qwen-API.txt',segmentation_prompt_mode='box_clip')
        cfg['m3']['input_image']=rel+'/m2/object_rgba.png'
        # Recorded resource adaptation after pathological million-face CNC UV charts.
        if key=='C_cnc':cfg['m3']['decimation_target']=250000
        cfg['m4']['local_background']['enabled']=False
        # This figure holds installation center fixed, rather than optimizing its translation.
        cfg['m4']['integration']['registration']['max_horizontal_fraction']=0.
        cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        (dest/'m1').symlink_to((PREP/'m1').resolve(),target_is_directory=True)
        shutil.copytree(PREP/'m2',dest/'m2')
        (dest/'records').mkdir(exist_ok=True)
        for f in ('trellis_model.json','trellis_dependencies.json'):
            shutil.copy2(BASE/'records'/f,dest/'records'/f)
        write_json(dest/'fixed_conditions.json',dict(source=str(PREP),variant=key,
            original_task_description=text,space_policy='Same current adaptive mask, camera and workstation volume for all variants',
            files={p.name:sha256(p) for p in (dest/'m2').iterdir() if p.is_file()},
            max_generation_requests=1,background_refinement=False))
    return Context(cp)

def m3_worker(config):
    ctx=Context(config)
    from industrial_swap_v2.core.pbr_pipeline import run
    start=time.monotonic();run(ctx)
    write_json(ctx.out/'m3_elapsed.json',dict(seconds=time.monotonic()-start))

def main():
    OUT.mkdir(parents=True,exist_ok=True);jobs=[];status={}
    for gpu,(key,(label,text)) in zip((1,2,3),VARIANTS.items()):
        try:
            ctx=prepare(key,text,gpu);prompt=build_prompt(ctx)
            (ctx.stage('m2')/'generation_prompt.txt').write_text(prompt)
            start=time.monotonic()
            generate(ctx,[ctx.stage('m2')/x for x in ('anchor.png','position_hint.png','depth_background.png')],prompt)
            write_json(ctx.out/'qwen_elapsed.json',dict(seconds=time.monotonic()-start))
            if not (ctx.stage('m2')/'object_rgba.png').exists():
                with (ctx.logs/'sam.log').open('a') as log:
                    subprocess.run([str(ctx.source(ctx.cfg['project']['sam_python'])),'-m',
                        'industrial_swap_v2.core.segmentation','--config',str(ctx.config_path)],cwd=ROOT,
                        env=ctx.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
            ctx.complete('m2',[ctx.stage('m2')/x for x in ('edited_scene.png','object_rgba.png','object_mask.png','anchor_camera.json')],fixed_condition_experiment=True)
            if (ctx.stage('m3')/'pbr_training_summary.json').exists():
                status[key]={'status':'m3_cached'};continue
            log=(ctx.logs/'m3.log').open('a')
            env=ctx.env();env['SWAP_GPU']=str(gpu)
            p=subprocess.Popen([sys.executable,__file__,'--m3',str(ctx.config_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((key,p,log));status[key]={'status':'m3_running','pid':p.pid,'gpu':gpu}
            print(key,'QWEN/SAM COMPLETE; M3 PID',p.pid,flush=True)
        except Exception as e:
            status[key]={'status':'failed_no_qwen_retry','error':str(e)};traceback.print_exc()
        write_json(OUT/'generation_status.json',status)
    for key,p,log in jobs:
        code=p.wait();log.close();status[key].update(status='m3_complete' if code==0 else 'm3_failed',exit_code=code)
        write_json(OUT/'generation_status.json',status);print(key,status[key],flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--m3':m3_worker(sys.argv[2])
    else:main()
