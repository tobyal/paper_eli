"""Check a live atomic DI-SDS checkpoint on CPU, without changing training."""
import argparse
import numpy as np
import torch
from ..core.context import Context, read_json, sha256, write_json
from ..core.scene import read_vertices


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
    ctx=Context(args.config); folder=ctx.stage('m4')
    checkpoint=torch.load(folder/'checkpoint_latest.pth',map_location='cpu',weights_only=False)
    source=read_vertices(ctx.stage('m3')/'object_world.ply')
    for buffer,names in [('xyz',['x','y','z']),('log_scale',[f'scale_{i}' for i in range(3)]),('rotation',[f'rot_{i}' for i in range(4)])]:
        expected=np.column_stack([source[n] for n in names]); observed=checkpoint['state'][buffer].numpy()
        assert expected.tobytes()==observed.tobytes(),buffer
    config=read_json(folder/'effective_config.json')
    assert sha256(config['background_source'])==config['background_sha256']
    result={'step':checkpoint['step'],'object_geometry_bitwise_preserved':True,
        'background_sha256_unchanged':True,'geometry_points':len(source),
        'mask_source':config['diffusion_mask_source'],'fingerprint_matches':checkpoint['fingerprint']==config['fingerprint']}
    assert result['fingerprint_matches']
    write_json(ctx.out/'validation/live_checkpoint_check.json',result); print(result)


if __name__=='__main__': main()
