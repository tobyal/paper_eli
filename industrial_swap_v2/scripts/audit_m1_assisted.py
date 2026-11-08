"""Read-only audit of an actual completed M1 case; no models or training."""
import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData
from ..core.context import read_json,write_json,sha256,Context


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    folder=a.case.resolve();ctx=Context(folder/'config.yaml');out=folder/'outputs/m1'
    integrity=read_json(folder/'integrity.json')
    for name,expected in integrity.items():assert sha256(folder/name)==expected,name
    receipt=read_json(folder/'records/m1_status.json')
    for artifact in receipt['artifacts']:assert sha256(artifact['path'])==artifact['sha256'],artifact['path']
    assert receipt['config_signature']==ctx.signature('m1')
    source_path=ctx.source(ctx.cfg['project']['source_ply']);assert sha256(source_path)==ctx.cfg['project']['source_sha256']
    original=PlyData.read(str(source_path))['vertex'].data
    ids=np.load(out/'target_indices.npy');keep=np.load(out/'keep_indices.npy')
    assert np.array_equal(np.sort(np.r_[ids,keep]),np.arange(len(original)))
    for name,index in [('target_gaussians.ply',ids),('background_gaussians.ply',keep)]:
        vertices=PlyData.read(str(out/name))['vertex'].data
        assert vertices.dtype==original.dtype
        for start in range(0,len(index),100000):assert vertices[start:start+100000].tobytes()==original[index[start:start+100000]].tobytes(),name
    result=read_json(folder/'result.json');assert result['executed_stages']==['m1'] and result['qwen_calls']==0 and result['trellis_calls']==0
    assert len({v['camera']['name'] for v in result['views']})==3
    assert not (folder/'outputs/m2').exists() and not (folder/'outputs/m3').exists() and not (folder/'outputs/m4').exists()
    selection=read_json(folder/'reference/selection.json');source_photo=ctx.root/'data/industrial/images'/selection['frame']
    assert sha256(source_photo)==sha256(folder/'reference/original_photo.jpg')==selection['hashes']['original_photo']
    assert (ctx.root/ctx.cfg['m1']['localization_dir']).resolve()==folder/'localization'
    report={'case':str(folder),'source_sha256':sha256(source_path),'all_saved_hashes_valid':True,'receipt_hashes_valid':True,
            'original_photo_byte_identical':True,'partition_indices_exact':True,'source_gaussian_parameters_byte_identical':True,
            'self_contained_m1_paths':True,'independent_views':3,'only_m1_executed':True,'qwen_calls':0,'trellis_calls':0,
            'counts':result['counts'],'quality':result['quality']}
    write_json(a.report,report);print(report)


if __name__=='__main__':main()
