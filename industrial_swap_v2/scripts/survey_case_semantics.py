"""Query real LangSplat fields and automatic SAM/CLIP instances for new cases.

Run in envs/gsle_langsplat; this stage cannot call any generation service.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy import ndimage


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('rank','fields','all'),default='all')
    parser.add_argument('--queries',type=Path);args=parser.parse_args()
    root=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli'); base=root/'industrial_swap_v2'
    out=args.output.resolve(); out.relative_to(base); out.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SWAP_DISABLE_QWEN='1')
    sys.path.insert(0,str(root/'repos/LangSplat')); import preprocess as official
    queries=json.loads((args.queries or base/'configs/case_queries.json').read_text())
    texts=[q['text'] for q in queries]+['factory floor']
    pins=json.loads((root/'records/gsle/models.json').read_text())
    clip=official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(Path(pins['openclip']['path'])/'open_clip_pytorch_model.bin'),positives=tuple(texts)))
    (out/'queries.json').write_text(json.dumps(queries,indent=2))
    if args.phase in ('rank','all'):
        records=[]; feature_dir=root/'data/gsle_langsplat/language_features'
        for number,path in enumerate(sorted(feature_dir.glob('*_f.npy'))):
            frame=path.stem[:-2]+'.jpg'
            features=torch.as_tensor(np.load(path),device='cuda',dtype=torch.float16)
            with torch.no_grad(): scores=torch.stack([clip.get_relevancy(features,i)[:,0] for i in range(len(queries))],1).cpu().numpy()
            segments=np.load(feature_dir/(path.stem[:-2]+'_s.npy')).astype(np.int32)
            height,width=segments.shape[1:]; frame_records=[]
            for level in (1,2,3):
                labels=segments[level]+1; sizes=np.bincount(labels.ravel(),minlength=len(features)+1)
                boxes=ndimage.find_objects(labels,max_label=len(features))
                for index,slices in enumerate(boxes):
                    if slices is None: continue
                    area=int(sizes[index+1]); fraction=area/(height*width)
                    ys,xs=slices
                    if not .003<fraction<.28 or min(xs.start,ys.start)<2 or xs.stop>=width-2 or ys.stop>=height-2: continue
                    cx=(xs.start+xs.stop)/2/width; cy=(ys.start+ys.stop)/2/height
                    center_weight=np.exp(-4*((cx-.5)**2+(cy-.5)**2))
                    for qi in np.flatnonzero(scores[index]>.5):
                        frame_records.append({'query_index':int(qi),'query_id':queries[qi]['id'],'text':queries[qi]['text'],
                            'frame':frame,'level':level,'feature_index':index,'area_fraction':fraction,
                            'semantic_score':float(scores[index,qi]),'rank':float(scores[index,qi]*center_weight),
                            'bbox':[xs.start,ys.start,xs.stop,ys.stop]})
            for qi in range(len(queries)):
                records.extend(sorted([r for r in frame_records if r['query_index']==qi],key=lambda r:r['rank'],reverse=True)[:3])
            if number%30==0: print('Ranked frames',number,'proposals',len(records),flush=True)
        records.sort(key=lambda r:r['rank'],reverse=True)
        (out/'reference_proposals.json').write_text(json.dumps(records,indent=2))
        for qi,query in enumerate(queries):
            chosen=[]
            for row in (r for r in records if r['query_index']==qi):
                if all(abs(int(row['frame'][6:12])-int(p['frame'][6:12]))>=12 for p in chosen): chosen.append(row)
                if len(chosen)==12: break
            canvas=Image.new('RGB',(3*470,4*294),'white'); draw=ImageDraw.Draw(canvas)
            for i,row in enumerate(chosen):
                image=np.asarray(Image.open(root/'data/industrial/images'/row['frame']).convert('RGB').resize((938,528))).copy()
                segments=np.load(feature_dir/(Path(row['frame']).stem+'_s.npy'))
                mask=segments[row['level']]==row['feature_index']; image[mask]=.4*image[mask]+.6*np.array([255,30,30])
                x=i%3*470;y=i//3*294; canvas.paste(Image.fromarray(image).resize((470,264)),(x,y+30))
                draw.text((x+4,y+4),f"{i}: {row['frame']} L{row['level']} id{row['feature_index']} {row['semantic_score']:.3f}",fill='black')
            canvas.save(out/(query['id']+'_proposals.jpg'),quality=92)
            (out/(query['id']+'_top.json')).write_text(json.dumps(chosen,indent=2))
        print('Reference survey saved',len(records),flush=True)
    if args.phase in ('fields','all'):
        sys.path.insert(0,str(root/'repos/LangSplat/autoencoder')); from model import Autoencoder
        decoder=Autoencoder([256,128,64,32,3],[16,32,64,128,256,256,512]).cuda().eval()
        decoder.load_state_dict(torch.load(root/'repos/LangSplat/autoencoder/ckpt/industrial_gsle/best_ckpt.pth'))
        original,_=torch.load(root/'outputs/reconstruction/chkpnt30000.pth',map_location='cpu')
        provenance=[]
        for level in (1,2,3):
            begin=time.time(); model,iteration=torch.load(root/f'outputs/gsle/langsplat_{level}/chkpnt30000.pth',map_location='cpu')
            assert iteration==30000 and len(model)==13
            for i in range(1,7): assert torch.equal(original[i],model[i]),('Source Gaussian order/parameters changed',level,i)
            field=model[7].detach(); scores=np.lib.format.open_memmap(out/f'scores_level_{level}.npy',mode='w+',dtype='float32',shape=(len(field),len(texts)))
            offset=0
            with torch.no_grad():
                for chunk in field.split(32768):
                    latent=chunk.cuda();latent=latent/(latent.norm(dim=-1,keepdim=True)+1e-9)
                    features=decoder.decode(latent);features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                    values=torch.stack([clip.get_relevancy(features,i)[:,0] for i in range(len(texts))],1).cpu().numpy()
                    scores[offset:offset+len(chunk)]=values;offset+=len(chunk)
            scores.flush(); provenance.append({'level':level,'iterations':iteration,'count':len(field),'source_parameters_identical':True,'seconds':time.time()-begin})
            print('Queried trained 3D language field',provenance[-1],flush=True); del model,field,scores
        (out/'semantic_provenance.json').write_text(json.dumps({'levels':provenance,'queries':texts,
            'normalization':'normalize latent before official autoencoder decoding; normalize decoded CLIP features',
            'scoring':'official OpenCLIP get_relevancy with generic negatives; same thresholds as gsle/localize.py',
            'fresh_semantic_queries':True,'old_target_indices_reused':False,'qwen_calls':0},indent=2))


if __name__=='__main__': main()
