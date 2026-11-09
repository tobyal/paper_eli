"""Create an isolated M4 registration run using completed M1/M2/PBR M3."""
import argparse
import copy
from pathlib import Path
import yaml
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--destination',required=True);p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--iterations',type=int,default=2000);a=p.parse_args()
    source=Context(a.source_config);out=source.path(a.destination)
    if out.exists():raise ValueError('Choose a new output directory; existing experiments are immutable')
    out.mkdir(parents=True);cfg=copy.deepcopy(source.cfg)
    for stage in ('m1','m2','m3'):(out/stage).symlink_to(source.stage(stage).resolve(),target_is_directory=True)
    cfg['project'].update(output_dir=a.destination,gpu=a.gpu,records_dir=a.destination+'/records',logs_dir=a.destination+'/logs')
    cfg['m4'].pop('resume',None)
    cfg['m4']['integration']={'alignment':'rasterized_reference','existing_transform':str(source.stage('m4')/'transform.json')}
    cfg['m4']['local_background']['iterations']=a.iterations;cfg['m4']['save_every']=500
    path=out/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8');ctx=Context(path)
    sources=[source.stage('m3')/'object_refined_local.ply',source.stage('m4')/'object_world.ply',
        source.stage('m4')/'transform.json',source.stage('m1')/'background_gaussians.ply',
        source.stage('m2')/'object_rgba.png',source.stage('m2')/'object_mask.png',
        source.stage('m2')/'anchor_camera.json']
    region=source.stage('m2')/'generation_region.json'
    if region.exists():sources.append(region)
    write_json(out/'source_preservation.json',{str(x):sha256(x) for x in sources})
    write_json(out/'provenance.json',{'source_config':str(a.source_config.resolve()),'previous_m4':str(source.stage('m4')),
        'm1_m2_m3_reused':True,'qwen_calls':0,'trellis_calls':0,'new_device_training_steps':0,'m4_steps':a.iterations})
    receipt=read_json(source.records/'m3_status.json')
    ctx.complete('m3',[v['path'] for v in receipt['artifacts']],reused_from=str(source.stage('m3')))
    print(path)


if __name__=='__main__':main()
