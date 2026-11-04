"""Run an official entry point, preserving its algorithm and recording resources."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ['ROOT'])
parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['gaussianeditor','dge','dreamcatalyst','editsplat'])
parser.add_argument('--output-tag',default='',help='Optional safe suffix for an independent replay; existing outputs are never overwritten')
args = parser.parse_args()
method = args.method
experiment = json.loads((root/'records/experiment.json').read_text())
prompt = experiment['instruction_en']
models = json.loads((root/'records/core_model_downloads.json').read_text())
ip2p_path = models['timbrooks/instruct-pix2pix']['path']
text_path = models['stable-diffusion-v1-5/stable-diffusion-v1-5']['path']
seg = 'white machine base'
data = root/'data/edit_train'
ply = root/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
base = root/'outputs/reconstruction/chkpnt30000.pth'
assert ply.exists() and data.exists(), 'Missing reconstructed source or common editing data'
assert (root/f'records/{method}.environment.completed').exists(), 'Environment is not verified'
repos = {'gaussianeditor':'GaussianEditor','dge':'DGE','dreamcatalyst':'DreamCatalyst/threestudio','editsplat':'cvpr25_EditSplat'}
repo = root/'repos'/repos[method]
python = root/f'envs/{method}/bin/python'
assert not args.output_tag or all(c.isalnum() or c in '_-' for c in args.output_tag), 'Unsafe output tag'
output_name=method+('_'+args.output_tag if args.output_tag else '')
destination = root/'outputs'/output_name
assert not destination.exists() or not any(destination.iterdir()), 'Output directory is not empty; use --output-tag for an independent run'
destination.mkdir(parents=True, exist_ok=True)
common = ['--train','--gpu','0', f'data.source={data}', f'system.gs_source={ply}', f'system.prompt_processor.prompt={prompt}', f'system.seg_prompt={seg}', f'exp_root_dir={destination}', 'tag=industrial_red_seed0', 'use_timestamp=false', 'seed=0', 'system.loggers.wandb.enable=false', f'system.cache_dir=industrial_red_{output_name}', 'trainer.val_check_interval=500']
common += ['system.mask_thres=0.5', f'system.guidance.ip2p_name_or_path={ip2p_path}', f'system.guidance.ddim_scheduler_name_or_path={models["CompVis/stable-diffusion-v1-4"]["path"]}']
if method == 'gaussianeditor':
    command = [str(python),'launch.py','--config','configs/edit-n2n.yaml'] + common + [
        'trainer.max_steps=2500', 'system.max_densify_percent=0.01',
        'system.anchor_weight_init_g0=0.01', 'system.anchor_weight_init=0.01', 'system.anchor_weight_multiplier=1.3',
        'system.gs_lr_scaler=0.1', 'system.gs_final_lr_scaler=0.1', 'system.color_lr_scaler=1',
        'system.opacity_lr_scaler=2', 'system.scaling_lr_scaler=2', 'system.rotation_lr_scaler=2',
        'system.loss.lambda_anchor_color=0', 'system.loss.lambda_anchor_geo=50',
        'system.loss.lambda_anchor_scale=50', 'system.loss.lambda_anchor_opacity=50',
        'system.densify_from_iter=100', 'system.densify_until_iter=2001',
        'system.edit_until_step=2200', 'system.densification_interval=100',
        f'system.prompt_processor.pretrained_model_name_or_path={text_path}']
elif method == 'dge':
    command = [str(python),'launch.py','--config','configs/dge.yaml'] + common + [
        f'system.prompt_processor.pretrained_model_name_or_path={text_path}']
elif method == 'dreamcatalyst':
    command = [str(python),'launch.py','--config','configs/edit-dc.yaml'] + common + [
        'trainer.max_steps=1500', f'system.dds_target_prompt_processor.prompt={prompt}',
        'system.dds_source_prompt_processor.prompt=A photo of a white industrial machine in a factory.',
        'system.max_densify_percent=0.01', 'system.anchor_weight_init_g0=0.05',
        'system.anchor_weight_init=0.1', 'system.anchor_weight_multiplier=1.3',
        'system.gs_lr_scaler=1', 'system.gs_final_lr_scaler=1', 'system.color_lr_scaler=1',
        'system.opacity_lr_scaler=1', 'system.scaling_lr_scaler=1', 'system.rotation_lr_scaler=1',
        'system.loss.lambda_anchor_color=5', 'system.loss.lambda_anchor_geo=5',
        'system.loss.lambda_anchor_scale=5', 'system.loss.lambda_anchor_opacity=5',
        'system.densify_from_iter=100', 'system.densify_until_iter=1201', 'system.densification_interval=100',
        'system.second_guidance.chi=0.075', 'system.second_guidance.delta=0.05', 'system.second_guidance.gamma=0.2',
        f'system.second_guidance.ip2p_name_or_path={ip2p_path}',
        f'system.second_guidance.ddim_scheduler_name_or_path={models["CompVis/stable-diffusion-v1-4"]["path"]}',
        f'system.prompt_processor.pretrained_model_name_or_path={text_path}',
        f'system.dds_target_prompt_processor.pretrained_model_name_or_path={text_path}',
        f'system.dds_source_prompt_processor.pretrained_model_name_or_path={text_path}']
else:
    data = root/'data/edit_all'
    reward=json.loads((root/'records/imagereward_model_downloads.json').read_text())['THUDM/ImageReward']['path']
    os.environ['IP2P_PATH']=ip2p_path
    os.environ['SD14_PATH']=models['CompVis/stable-diffusion-v1-4']['path']
    os.environ['IMAGEREWARD_PATH']=str(Path(reward)/'ImageReward.pt')
    os.environ['IMAGEREWARD_CONFIG']=str(Path(reward)/'med_config.json')
    command = [str(python),'run_editing.py','-s',str(data),'-m',str(destination),
        '--source_checkpoint',str(base),'--object_prompt','machine','--target_prompt',prompt,
        '--sampling_prompt','A factory with the central industrial machine housing painted red and all other objects unchanged.',
        '--target_mask_prompt',seg,'-r','1']

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run_dir = root/'records/runs'/f'{method}_{stamp}'
run_dir.mkdir(parents=True)
os.environ['PYTHONPATH']=str(root/'scripts')+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['TARGET_GEOMETRY']=str(root/'records/target_geometry.json')
os.environ['INSTANCE_SELECTION_LOG']=str(run_dir/'instance_selection.jsonl')
# All required official snapshots were downloaded and verified before execution.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
run = {'method':method, 'cwd':str(repo),'argv':command,'start_utc':datetime.now(timezone.utc).isoformat(),
    'output_name':output_name,'editing_data':str(data),'instruction':prompt,
    'hf_runtime_offline':True,
    'log':str(run_dir/'stdout.log'), 'source_ply':str(ply), 'status':'running',
    'shared_gpu':True, 'gpu':os.environ.get('CUDA_VISIBLE_DEVICES','0'),
    'sampled_process_gpu_peak_mib':0,'gpu_sampling_period_seconds':5,
    'scene_adaptation':'Common physical-instance selection among unchanged LangSAM candidates using reviewed projected 3D ROI; no substitute masks or pixel recoloring',
    'segmentation_prompt':seg,
    'semantic_tracing_threshold':0.5 if method!='editsplat' else None,
    'timing_scope':'Entire official entrypoint including model loading, segmentation, editing, fitting and its built-in rendering'}
(run_dir/'run.json').write_text(json.dumps(run,indent=2))
with (run_dir/'stdout.log').open('w') as log, (run_dir/'gpu.csv').open('w') as gpu:
    gpu.write('elapsed_s,index,utilization_gpu_percent,memory_used_MiB,memory_free_MiB\n')
    start=time.monotonic()
    process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    run['pid']=process.pid
    (run_dir/'run.json').write_text(json.dumps(run,indent=2))
    while process.poll() is None:
        sample=subprocess.run(['nvidia-smi','--id='+run['gpu'],'--query-gpu=index,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True)
        if sample.returncode == 0:
            gpu.write(f'{time.monotonic()-start:.3f},'+sample.stdout.strip()+'\n')
            gpu.flush()
        memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
        if memory.returncode==0:
            for line in memory.stdout.splitlines():
                fields=[field.strip() for field in line.split(',')]
                if len(fields)==2 and fields[0]==str(process.pid) and fields[1].isdigit():
                    run['sampled_process_gpu_peak_mib']=max(run['sampled_process_gpu_peak_mib'],int(fields[1]))
        time.sleep(5)
    run['wall_seconds']=time.monotonic()-start
    run['exit_code']=process.returncode
run['end_utc']=datetime.now(timezone.utc).isoformat()
run['status']='entrypoint_completed_needs_output_review' if run['exit_code']==0 else 'failed'
(run_dir/'run.json').write_text(json.dumps(run,indent=2))
print(json.dumps(run,indent=2),flush=True)
raise SystemExit(run['exit_code'])
