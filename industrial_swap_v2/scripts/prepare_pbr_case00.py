"""Create a separate case00 experiment using its existing M1/M2 and PBR GLB."""
import argparse
import shutil
from pathlib import Path
import yaml
from industrial_swap_v2.core.context import read_json,write_json,sha256
from industrial_swap_v2.webapp import store as s,service as v


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=2)
    args=p.parse_args();root=s.BASE/args.output;root.resolve().relative_to(s.BASE/'outputs')
    if root.exists(): raise ValueError('Use a fresh case00 output directory')
    root.mkdir(parents=True)
    source='9b0eeee5840d453e8f21e1763544a6cb';heads=s.unpack(v.experiment(source)['heads'])
    native=v.stage_dir('173dd1e87749494db5f66ab1287d37d7')
    cfg=yaml.safe_load((s.BASE/'configs/default.yaml').read_text())
    for stage in ('m1','m2'):
        (root/stage).symlink_to(v.stage_dir(heads[stage]),target_is_directory=True)
    cfg['project'].update(output_dir=args.output,records_dir=args.output+'/records',logs_dir=args.output+'/logs',gpu=args.gpu)
    cfg['m3']['input_image']=args.output+'/m2/object_rgba.png'
    cfg['m3']['pbr']['source_glb']=str(native/'trellis_asset.glb')
    # Same original GLB and frame; use the already validated case00 placement
    # for a controlled comparison. Other devices default to automatic fitting.
    cfg['m4']['integration'].update(alignment='verified_transform',existing_transform=str(native/'transform.json'))
    cfg['m4']['local_background']['iterations']=250
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    originals=[native/'trellis_asset.glb',native/'object_world.ply',native/'transform.json',
        root/'m1/background_gaussians.ply',root/'m1/target_gaussians.ply',root/'m2/object_rgba.png',
        root/'m2/object_mask.png',root/'m2/anchor_camera.json',s.ROOT/cfg['m3']['pbr']['hdri']]
    write_json(root/'source_preservation.json',{str(x.resolve()):sha256(x) for x in originals})
    write_json(root/'provenance.json',{'case':'case00','source_experiment':source,'source_m3':str(native),
        'source_heads':heads,'new_trellis_calls':0,'qwen_calls':0,'prior_results_overwritten':False})
    for name in ('records','logs'): (root/name).mkdir()
    for name in ('trellis_model.json','trellis_dependencies.json'):
        shutil.copy2(s.BASE/'records'/name,root/'records'/name)
    print(root/'config.yaml',flush=True)


if __name__=='__main__':main()
