"""Fixed seed/text/camera, four image-conditioning controls; one POST each."""
import sys,time,shutil,subprocess,traceback
from pathlib import Path
import yaml
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,write_json,sha256
from industrial_swap_v2.core.qwen_client import generate
BASE=ROOT/'industrial_swap_v2';OUT=BASE/'outputs/paper_figures_v4/F11_m2_ablation'
PREP=BASE/'outputs/paper_figures_v3/F09_text_variants/qwen_direct_adaptive_v2/preparation'

def prompt(cfg,key):
    text='Edit image 1 and return one photorealistic RGB image of the same factory from exactly the same camera. Image 1 is the original scene. '
    text+='Replace the entire old device: '+cfg['m1']['target_text']+'. '
    if key!='C1':text+='Image 2 is ONLY a position guide: the translucent magenta mask marks the allowed local replacement region, with protected holes for foreground equipment. Remove all magenta markings from the output. '
    if key=='C3':text+='Image 3 is ONLY the original scene depth reference, including the old device depth. Use it for perspective and scene context; do not copy its grayscale appearance. '
    if key=='C4':text+='Image 3 is ONLY the background depth reference. The editing area is blacked out as UNKNOWN depth, not a surface to copy. Use surrounding depth for perspective and scene context; do not copy its grayscale appearance. '
    text+='Generate exactly one complete replacement device: '+cfg['m2']['replacement_text']+' '
    text+='Place its base on the floor at the old device location and keep it within the original workstation footprint, with no floating or intersecting parts. Preserve all surrounding equipment, containers, railings, floor markings, lighting, perspective and foreground occlusions. Do not add extra equipment, people, annotations or a new background. Return the edited full factory view, not a collage, isolated product photo or depth map.'
    return text

def main():
    OUT.mkdir(parents=True,exist_ok=True);status={}
    for key in ['C1','C2','C3','C4']:
        dest=OUT/'conditions'/key;dest.mkdir(parents=True,exist_ok=True);cp=dest/'config.yaml'
        if not cp.exists():
            cfg=yaml.safe_load((PREP/'config.yaml').read_text());rel=str(dest.relative_to(BASE))
            cfg['project'].update(output_dir=rel,records_dir=rel+'/records',logs_dir=rel+'/logs',gpu=0)
            cfg['m2'].update(preparation_only=False,max_api_requests=1,api_key_file='/mnt/sda/ubuntu/lkx/papers/qwen-API.txt')
            cp.write_text(yaml.safe_dump(cfg,sort_keys=False));shutil.copytree(PREP/'m2',dest/'m2')
        ctx=Context(cp);p=prompt(ctx.cfg,key);(ctx.stage('m2')/'generation_prompt.txt').write_text(p)
        names=['anchor.png']+(['position_hint.png'] if key!='C1' else [])+(['depth.png'] if key=='C3' else ['depth_background.png'] if key=='C4' else [])
        try:
            start=time.monotonic();generate(ctx,[ctx.stage('m2')/x for x in names],p)
            status[key]=dict(status='generated',seconds=time.monotonic()-start,images=names,seed=ctx.cfg['m2']['seed'])
            print(key,'GENERATED',flush=True)
        except Exception as e:
            status[key]=dict(status='failed_no_retry',error=str(e));traceback.print_exc()
        write_json(OUT/'generation_status.json',status)
    write_json(OUT/'condition_policy.json',dict(shared_input_source=str(PREP),current_mask_sha256=sha256(PREP/'m2/edit_mask.png'),
        paired_seed=42,samples_per_condition=1,old_cache_excluded='Legacy C4 used a different mask; all four conditions use the updated mask/camera for a matched comparison.',
        scope='Two-dimensional generation only; no 3D generation or optimization'))

if __name__=='__main__':main()
