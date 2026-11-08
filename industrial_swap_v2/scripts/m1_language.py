"""Frozen official OpenCLIP inference: vocabulary cache or a fresh 3D query."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from ..core.context import read_json, write_json, sha256
from ..core.m1_reference import vocabulary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path)
    p.add_argument('--text');a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    out.relative_to(root/'industrial_swap_v2');out.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SWAP_DISABLE_QWEN='1')
    import torch
    sys.path.insert(0,str(root/'repos/LangSplat'));import preprocess as official
    library=root/'industrial_swap_v2/configs/m1_semantic_library.json'
    texts=[a.text,'factory floor'] if a.selection else vocabulary(library)
    pins=read_json(root/'records/gsle/models.json')
    weights=Path(pins['openclip']['path'])/'open_clip_pytorch_model.bin'
    clip=official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(clip_model_pretrained=str(weights),positives=tuple(texts)))
    meta={'model':'ViT-B-16','pretrained':'laion2b_s34b_b88k','weights':str(weights),'weights_sha256':sha256(weights),
          'generic_negatives':list(clip.config.negatives) if hasattr(clip,'config') else ['object','things','stuff','texture'],
          'library_sha256':sha256(library),'texts':texts,'retrained':False,'qwen_calls':0}
    if not a.selection:
        np.savez(out/'text_embeddings.npz',texts=np.array(texts),embeddings=clip.pos_embeds.detach().cpu().float().numpy(),
                 negatives=clip.neg_embeds.detach().cpu().float().numpy())
        write_json(out/'manifest.json',meta);print('Vocabulary encoded',len(texts),flush=True);return
    if not a.text:raise ValueError('A selected English description is required')
    prototype=torch.as_tensor(np.load(a.selection/'clip_prototype.npy'),device='cuda')
    sys.path.insert(0,str(root/'repos/LangSplat/autoencoder'));from model import Autoencoder
    decoder=Autoencoder([256,128,64,32,3],[16,32,64,128,256,256,512]).cuda().eval()
    decoder_path=root/'repos/LangSplat/autoencoder/ckpt/industrial_gsle/best_ckpt.pth'
    decoder.load_state_dict(torch.load(decoder_path));meta['decoder_sha256']=sha256(decoder_path)
    original,_=torch.load(root/'outputs/reconstruction/chkpnt30000.pth',map_location='cpu');meta['levels']=[]
    for level in (1,2,3):
        checkpoint=root/f'outputs/gsle/langsplat_{level}/chkpnt30000.pth'
        model,iteration=torch.load(checkpoint,map_location='cpu')
        assert iteration==30000 and len(model)==13
        for i in range(1,7):assert torch.equal(original[i],model[i]),('Source order/geometry mismatch',level,i)
        field=model[7].detach();scores=np.lib.format.open_memmap(out/f'scores_level_{level}.npy',mode='w+',dtype='float32',shape=(len(field),3))
        offset=0
        with torch.no_grad():
            for chunk in field.split(32768):
                latent=chunk.cuda();latent=latent/(latent.norm(dim=-1,keepdim=True)+1e-9)
                features=decoder.decode(latent);features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                values=torch.stack([clip.get_relevancy(features,i)[:,0] for i in (0,1)]+[features@prototype],1).cpu().numpy()
                scores[offset:offset+len(chunk)]=values;offset+=len(chunk)
        scores.flush();del scores,model,field
        meta['levels'].append({'level':level,'iteration':iteration,'source_parameters_identical':True,'checkpoint':str(checkpoint),'sha256':sha256(checkpoint)})
        print('Queried LangSplat level',level,flush=True)
    meta.update(columns=['text_relevance','floor_relevance','reference_clip_cosine_diagnostic_only'],
                normalization='normalize latent; official autoencoder decode; normalize decoded CLIP',fresh_query=True)
    write_json(out/'semantic_provenance.json',meta)


if __name__=='__main__':main()
