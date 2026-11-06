"""Create an isolated Web experiment using an unchanged existing world PLY."""
import argparse
import json
from pathlib import Path
from . import store as s, service as v


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-experiment',required=True)
    p.add_argument('--name',default='OAGOF · multiview guard · unchanged device')
    args=p.parse_args(); source=v.experiment(args.source_experiment)
    heads=s.unpack(source['heads']); m3=v.run(heads['m3'])
    assets=s.unpack(m3['outputs'])['assets']; world_asset=assets['object_world.ply']
    record={'source_experiment':source['id'],'source_heads':heads,'source_m3':m3['id'],
        'world_asset':world_asset,'world_sha256':s.asset(world_asset)['sha256']}
    experiment=v.create(args.name,copy_from=source['id'])
    v.select(experiment,world_asset,'m3')
    run=v.submit(experiment,'m3','reuse',s.uid(),parameters={'realign':False,
        'object_occlusion_filter':{'enabled':True,'use_multi_view':True,'preserve_object_geometry':True,'floor_repair':False}})
    record.update(experiment=experiment,run=run)
    (s.BASE/'records/oagof_web_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
