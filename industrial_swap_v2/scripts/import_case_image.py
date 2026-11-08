"""Later human-image handoff: finish M2 from a local image, never invoke Qwen."""
import argparse
import os
from pathlib import Path
from ..core.context import Context,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--kind',choices=('edited_scene','object_rgba'),default='edited_scene');args=parser.parse_args()
    ctx=Context(args.config);os.environ.update(ctx.env());os.environ['SWAP_DISABLE_QWEN']='1'
    out=ctx.stage('m2')
    if not (out/'conditions.json').exists():raise FileNotFoundError('Prepare case conditions first')
    if (out/'generation.json').exists() or (out/'edited_scene.png').exists() or (out/'object_rgba.png').exists():
        raise FileExistsError('M2 image already exists; preserve this experiment and use a new version')
    source=args.image.resolve()
    from ..webapp.adapter import image_import
    image_import(ctx,{'asset':{'path':str(source),'kind':args.kind,'sha256':sha256(source),'id':'manual-'+sha256(source)[:16]}})
    print('Manual image imported; M2 complete. M3 and M4 have not been started.',flush=True)


if __name__=='__main__':main()
