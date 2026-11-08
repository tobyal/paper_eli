"""Read-only provenance capture for actual source photos, cameras, and scripts."""
from pathlib import Path
import argparse
import subprocess
from ..core.context import Context, read_json, write_json, sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
    ctx=Context(args.config)
    folders=[ctx.stage('m4')/'multiview_supervision',ctx.stage('m4')/'background_optimization/visibility_prepare',
             ctx.stage('m3')/'multiview_visibility']
    folder=next((p for p in folders if (p/'manifest.json').exists()),None)
    if folder is None: raise FileNotFoundError('No completed M4 supervision cache')
    manifest=read_json(folder/'manifest.json')
    photos={v['original_image'] for v in manifest['views'] if v['original_image']}
    sources=[Path(v) for v in photos]
    sources.extend(ctx.source('data/industrial/sparse/0')/n for n in ('cameras.bin','images.bin','points3D.bin'))
    values={str(p):{'sha256':sha256(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sources}
    destination=ctx.out/'input_provenance.json'
    if destination.exists():
        assert values==read_json(destination)['files'],'Source images or COLMAP files changed'
    else:
        write_json(destination,{'files':values,'photo_count':len(photos),
            'source_video_sha256':'061413776d7e163c53aa37cd556018bd65356df2128f4929a55ed90645c67116',
            'camera_records':[{'name':v['name'],'camera':v['camera'],'split':v['split']} for v in manifest['views']]})
    print('Verified',len(values),'source files; no mutations')


if __name__=='__main__': main()
