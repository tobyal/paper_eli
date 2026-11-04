"""Compare real renders on held-out cameras using a fixed reviewed target mask.

There is no edited ground truth. Preservation compares to the common unedited
render. Reprojection uses source geometry and source visibility for every method.
"""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F
import clip
import lpips

root=Path(os.environ['ROOT'])
parser=argparse.ArgumentParser()
parser.add_argument('--methods',nargs='*',default=['gaussianeditor','dge','dreamcatalyst','editsplat'])
args=parser.parse_args()
base=root/'outputs/reconstruction/common_render'
mask_dir=root/'outputs/evaluation_masks/wide'
assert (root/'records/evaluation_masks.reviewed.json').exists(), 'Review masks visually before scoring'
cameras=json.loads((base/'cameras.json').read_text())
test=[cam for cam in cameras if cam['split']=='test']
assert len(test)==44
clip_path=json.loads((root/'records/clip_model.json').read_text())['path']
clip_model,clip_preprocess=clip.load(clip_path,device='cuda')
clip_model.eval().requires_grad_(False)
perceptual=lpips.LPIPS(net='vgg',spatial=True).cuda().eval().requires_grad_(False)
texts=['A photo of an industrial machine with a white housing in a factory.',
       'A photo of an industrial machine with a red housing in a factory.']
with torch.no_grad():
    text=F.normalize(clip_model.encode_text(clip.tokenize(texts).cuda()).float(),dim=-1)
    text_direction=F.normalize(text[1]-text[0],dim=-1)

def rgb(path): return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
def tensor(array): return torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).cuda()
def red_fraction(array,mask):
    hsv=cv2.cvtColor(array,cv2.COLOR_RGB2HSV)
    red=((hsv[:,:,0]<24)|(hsv[:,:,0]>336))&(hsv[:,:,1]>.3)&(hsv[:,:,2]>.15)
    return float(red[mask].mean())
def embed_crop(array,mask):
    yy,xx=np.where(mask)
    pad=20
    crop=array[max(0,yy.min()-pad):min(len(array),yy.max()+pad+1),max(0,xx.min()-pad):min(array.shape[1],xx.max()+pad+1)]
    image=Image.fromarray(np.round(crop*255).astype(np.uint8))
    return F.normalize(clip_model.encode_image(clip_preprocess(image).unsqueeze(0).cuda()).float(),dim=-1)[0]

def correspondence(a,b):
    patha=base/'wide'/Path(a['name']).with_suffix('.npz')
    pathb=base/'wide'/Path(b['name']).with_suffix('.npz')
    da,db=np.load(patha),np.load(pathb)
    z=da['depth'].astype(np.float32)
    yy,xx=np.indices(z.shape,dtype=np.float32)
    points=np.stack([(xx-a['cx'])/a['fx']*z,(yy-a['cy'])/a['fy']*z,z],axis=-1)
    world=(points-np.array(a['t']))@np.array(a['R'])
    target=world@np.array(b['R']).T+np.array(b['t'])
    zz=target[:,:,2]
    u=(b['fx']*target[:,:,0]/np.maximum(zz,1e-6)+b['cx']).astype(np.float32)
    v=(b['fy']*target[:,:,1]/np.maximum(zz,1e-6)+b['cy']).astype(np.float32)
    depth_b=cv2.remap(db['depth'].astype(np.float32),u,v,cv2.INTER_LINEAR)
    alpha_b=cv2.remap(db['alpha'].astype(np.float32),u,v,cv2.INTER_LINEAR)
    valid=(da['alpha']>.95)&(alpha_b>.95)&(z>0)&(zz>0)&(u>=1)&(u<b['width']-2)&(v>=1)&(v<b['height']-2)&(np.abs(depth_b-zz)<.03*np.maximum(zz,1e-6))
    return u,v,valid

def aggregate(rows):
    result={}
    for key in sorted({key for row in rows for key in row if key not in ['frame','pair']}):
        values=[row[key] for row in rows if isinstance(row.get(key),(float,int)) and np.isfinite(row[key])]
        if values: result[key]={'mean':float(np.mean(values)),'std':float(np.std(values)),'n':len(values)}
    return result

results={'protocol':{'test_views':len(test),'reference':'Common original 3DGS render, no edited ground truth',
 'mask':'Reviewed common physical target, with 9-pixel dilation for background exclusion',
 'clip':'OpenAI ViT-L/14; local target bounding box with 20px context',
 'reprojection':'Adjacent held-out source cameras; source alpha>.95 and relative depth agreement<3%; common source geometry for all methods',
 'lpips':'VGG spatial LPIPS, background pixels outside dilated mask',
 'aggregation':'Unweighted mean over valid views/pairs; n reported'},'methods':{}}
with torch.no_grad():
  for method in ['reconstruction']+args.methods:
    destination=root/'outputs'/method/'common_render/wide'
    rows=[]
    for cam in test:
        name=Path(cam['name']).with_suffix('.png')
        original=rgb(base/'wide'/name)
        edited=rgb(destination/name)
        mask=np.array(Image.open(mask_dir/name))>127
        assert mask.shape==original.shape[:2]
        background=~binary_dilation(mask,iterations=9)
        mse=float(((edited-original)**2)[background].mean())
        _,ssim_map=structural_similarity(original,edited,channel_axis=2,data_range=1.,full=True)
        if ssim_map.ndim==3: ssim_map=ssim_map.mean(axis=2)
        lp=perceptual(tensor(original)*2-1,tensor(edited)*2-1).squeeze().cpu().numpy()
        row={'frame':str(name),'target_pixels':int(mask.sum()),'background_mse':mse,
             'background_psnr_db':float(-10*np.log10(mse)) if mse>0 else None,
             'background_ssim':float(ssim_map[background].mean()),
             'background_lpips':float(lp[background].mean()),
             'background_mean_absolute_change':float(np.abs(edited-original)[background].mean())}
        if mask.sum()>=100:
            a,b=embed_crop(original,mask),embed_crop(edited,mask)
            change=b-a
            row.update(target_red_fraction=red_fraction(edited,mask),target_red_fraction_original=red_fraction(original,mask),clip_target_similarity=float(b@text[1]),clip_image_preservation=float(a@b))
            if float(change.norm())>1e-6:
                row['clip_directional_similarity']=float(F.normalize(change,dim=0)@text_direction)
        rows.append(row)
    pairs=[]
    for a,b in zip(test[:-1],test[1:]):
        u,v,valid=correspondence(a,b)
        na,nb=Path(a['name']).with_suffix('.png'),Path(b['name']).with_suffix('.png')
        ea,eb=rgb(destination/na),rgb(destination/nb)
        oa,ob=rgb(base/'wide'/na),rgb(base/'wide'/nb)
        error=np.abs(ea-cv2.remap(eb,u,v,cv2.INTER_LINEAR)).mean(axis=2)
        source_error=np.abs(oa-cv2.remap(ob,u,v,cv2.INTER_LINEAR)).mean(axis=2)
        mask=np.array(Image.open(mask_dir/na))>127
        row={'pair':[str(na),str(nb)],'valid_pixels':int(valid.sum())}
        for label,selection in [('all',valid),('target',valid&mask),('background',valid&~binary_dilation(mask,iterations=9))]:
            if selection.sum()<100: continue
            row[label+'_reprojection_mae']=float(error[selection].mean())
            row[label+'_reprojection_excess_mae']=float((error-source_error)[selection].mean())
        pairs.append(row)
    results['methods'][method]={'views':rows,'pairs':pairs,'view_aggregate':aggregate(rows),'visible_target_view_aggregate':aggregate([row for row in rows if row['target_pixels']>=100]),'pair_aggregate':aggregate(pairs)}
    (root/'records/common_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(method,json.dumps(results['methods'][method]['view_aggregate']),flush=True)
