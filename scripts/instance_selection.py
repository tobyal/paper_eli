"""Select a physical instance among unmodified LangSAM candidate masks.

The projected annotation only ranks candidates. It never creates a substitute
segmentation mask, recolors pixels, or changes any editing/fitting algorithm.
"""
import itertools
import json
import os
from pathlib import Path
import numpy as np
import torch

def projected_target_box(camera):
    filename=os.environ.get('TARGET_GEOMETRY')
    if not filename: return None
    geometry=json.loads(Path(filename).read_text())
    signs=np.array(list(itertools.product([-1,1],repeat=3)))
    xyz=np.array(geometry['center'])+(signs*np.array(geometry['half_extents']))@np.array(geometry['basis'])
    homogeneous=np.column_stack([xyz,np.ones(8)])
    matrix=camera.full_proj_transform.detach().cpu().numpy()
    projected=homogeneous@matrix
    if np.any(projected[:,3]<=0): return []
    xy=(projected[:,:2]/projected[:,3:4]+1)*np.array([camera.image_width,camera.image_height])/2
    low=np.maximum(np.floor(xy.min(axis=0)),0)
    high=np.minimum(np.ceil(xy.max(axis=0)),[camera.image_width,camera.image_height])
    if np.any(high-low<2): return []
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]

def select_instance(masks,camera):
    box=projected_target_box(camera)
    height,width=int(camera.image_height),int(camera.image_width)
    if masks.ndim==2: masks=masks.unsqueeze(0)
    zero=torch.zeros((1,height,width),device=masks.device,dtype=torch.float32)
    if box is None:
        return masks[:1].float() if masks.ndim==3 and len(masks) else zero
    selected=-1
    scores=[]
    if len(box)==4 and masks.ndim==3 and len(masks):
        x1,y1,x2,y2=box
        roi_area=(x2-x1)*(y2-y1)
        candidate_area=masks.float().sum(dim=(-2,-1))
        overlap=masks[:,y1:y2,x1:x2].float().sum(dim=(-2,-1))
        # Binary cosine overlap favors a spatially matching component over a
        # large whole-image mask, without cropping the chosen semantic mask.
        score=overlap/torch.sqrt(candidate_area.clamp_min(1)*roi_area)
        scores=score.detach().cpu().tolist()
        best=int(score.argmax())
        if float(score[best])>=.25: selected=best
    result=masks[selected:selected+1].float() if selected>=0 else zero
    logfile=os.environ.get('INSTANCE_SELECTION_LOG')
    if logfile:
        with open(logfile,'a') as stream:
            stream.write(json.dumps({'camera':getattr(camera,'image_name',None),'projected_roi':box,'candidate_count':len(masks),'scores':scores,'selected_index':selected,'selected_pixels':int(result.sum())})+'\n')
    return result
