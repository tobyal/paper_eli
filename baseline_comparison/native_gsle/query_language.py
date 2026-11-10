"""Query all trained 3D language fields using only A and generic floor text."""
import json,os,sys,time
from pathlib import Path
import numpy as np,torch

root=Path(os.environ['ROOT']);out=Path(os.environ['GSLE_OUTPUT']);dest=out/'localization'
cfg=json.loads(Path(os.environ['GSLE_CONFIG']).read_text())
assert not any(k.startswith('reference_') for k in cfg)
sys.path.insert(0,str(root/'repos/LangSplat'));import preprocess as official
sys.path.insert(0,str(root/'repos/LangSplat/autoencoder'));from model import Autoencoder
pins=json.loads((root/'records/gsle/models.json').read_text())
clip=official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(
    clip_model_pretrained=str(Path(pins['openclip']['path'])/'open_clip_pytorch_model.bin'),positives=(cfg['target_text'],'factory floor')))
decoder=Autoencoder([256,128,64,32,3],[16,32,64,128,256,256,512]).cuda().eval()
decoder.load_state_dict(torch.load(root/'repos/LangSplat/autoencoder/ckpt/industrial_gsle/best_ckpt.pth'))
source,_=torch.load(root/'outputs/reconstruction/chkpnt30000.pth',map_location='cpu')
records=[];start=time.time()
for level in (1,2,3):
    model,iteration=torch.load(root/f'outputs/gsle/langsplat_{level}/chkpnt30000.pth',map_location='cpu')
    assert iteration==30000 and len(model)==13
    for index in range(1,7):assert torch.equal(source[index],model[index]),('Source geometry differs',level,index)
    scores=[];floor=[]
    with torch.no_grad():
        for chunk in model[7].detach().split(32768):
            latent=chunk.cuda();latent=latent/(latent.norm(dim=-1,keepdim=True)+1e-9)
            f=decoder.decode(latent);f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            scores.append(clip.get_relevancy(f,0)[:,0].cpu().numpy())
            floor.append(clip.get_relevancy(f,1)[:,0].cpu().numpy())
    scores=np.concatenate(scores);floor=np.concatenate(floor)
    np.savez_compressed(dest/f'language_query_level{level}.npz',scores=scores,floor_scores=floor)
    records.append(dict(level=level,iterations=iteration,gaussians=len(scores),score_quantiles=np.quantile(scores,[0,.5,.9,.99,1]).tolist(),positive_points=int((scores>.5).sum()),source_parameters_identical=True))
    print(records[-1],flush=True);del model
(dest/'query.json').write_text(json.dumps(dict(text=cfg['target_text'],levels=records,elapsed=time.time()-start,
    method='Official LangSplat pairwise text relevance against generic negatives, temperature 10, worst negative; all three trained 3D fields',
    reference_image=None,prototype=None,manual_or_ours_region_used=False),indent=2))
