"""CPU access to original COLMAP photos and existing SAM/CLIP region caches."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import numpy as np


def vocabulary(path):
    library=json.loads(Path(path).read_text(encoding='utf-8'))
    texts=list(library['descriptions'])
    for category in library['categories']:
        article='an' if category[0].lower() in 'aeiou' else 'a'
        texts.append(f'{article} {category}')
        texts.extend(f'a {color} {category}' for color in library['colors'])
    return list(dict.fromkeys(texts))


@lru_cache(maxsize=2)
def cameras(root):
    root=Path(root)
    spec=importlib.util.spec_from_file_location('m1_colmap_loader',root/'repos/gaussian-splatting/scene/colmap_loader.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sparse=root/'data/industrial/sparse/0'
    intr=module.read_intrinsics_binary(str(sparse/'cameras.bin'))
    ext=module.read_extrinsics_binary(str(sparse/'images.bin'))
    result={}
    for im in sorted(ext.values(),key=lambda item:item.name):
        calibration=intr[im.camera_id]
        if calibration.model!='PINHOLE':raise ValueError('Expected existing undistorted PINHOLE calibration')
        width=938;height=2*round(calibration.height*width/calibration.width/2)
        fx,fy,cx,cy=calibration.params
        result[im.name]={'R':module.qvec2rotmat(im.qvec).tolist(),'t':im.tvec.tolist(),
            'width':width,'height':height,'fx':fx*width/calibration.width,'fy':fy*height/calibration.height,
            'cx':cx*width/calibration.width,'cy':cy*height/calibration.height,'name':im.name}
    return result


def frame_paths(root,frame):
    if frame not in cameras(str(root)):raise ValueError('照片未在 COLMAP 中配准')
    root=Path(root);prefix=root/'data/gsle_langsplat/language_features'/Path(frame).stem
    return root/'data/industrial/images'/frame,Path(str(prefix)+'_s.npy'),Path(str(prefix)+'_f.npy')


@lru_cache(maxsize=8)
def load_arrays(segments_path,features_path,segment_stamp,feature_stamp):
    return np.load(segments_path,mmap_mode='r'),np.load(features_path,mmap_mode='r')


def frame_arrays(root,frame):
    photo,sp,fp=frame_paths(root,frame)
    if not sp.is_file() or not fp.is_file():raise ValueError('该配准照片缺少 SAM/CLIP 缓存，请选择标有可用缓存的照片')
    seg,features=load_arrays(str(sp),str(fp),sp.stat().st_mtime_ns,fp.stat().st_mtime_ns)
    camera=cameras(str(root))[frame]
    if seg.shape[1:]!=(camera['height'],camera['width']) or features.ndim!=2 or features.shape[1]!=512:
        raise ValueError('缓存与相机或 CLIP 维度不匹配')
    return seg,features


def selected_region(segments,features,level,ids):
    if level not in (1,2,3):raise ValueError('请选择 SAM 尺度 1、2 或 3')
    ids=sorted(set(int(i) for i in ids))
    if not 1<=len(ids)<=64:raise ValueError('请选择 1–64 个同尺度 SAM 区域')
    labels=segments[level]
    if any(i<0 or i>=len(features) or not np.any(labels==i) for i in ids):
        raise ValueError('SAM 区域不存在或不属于当前照片/尺度')
    mask=np.isin(labels,ids)
    counts=np.asarray([(labels==i).sum() for i in ids],dtype=np.float64)
    original=np.asarray(features[ids],dtype=np.float32)
    if not np.isfinite(original).all():raise ValueError('CLIP 缓存包含非有限值')
    normalized=original/np.maximum(np.linalg.norm(original,axis=1,keepdims=True),1e-8)
    weights=counts/counts.sum();prototype=(normalized*weights[:,None]).sum(0)
    norm=np.linalg.norm(prototype)
    if norm<1e-8:raise ValueError('选中区域的 CLIP 特征无效')
    return mask,original,weights,(prototype/norm).astype(np.float32)


def rank_descriptions(prototype,texts,embeddings,negatives):
    prototype=np.asarray(prototype,dtype=np.float32)
    prototype=prototype/max(float(np.linalg.norm(prototype)),1e-8)
    scores=np.asarray(embeddings,dtype=np.float32)@prototype
    worst_negative=float(np.max(np.asarray(negatives,dtype=np.float32)@prototype))
    relevance=1/(1+np.exp(np.clip(10*(worst_negative-scores),-80,80)))
    order=np.argsort(-scores,kind='stable')
    return [{'rank':rank+1,'text':str(texts[i]),'cosine_similarity':float(scores[i]),
        'langsplat_relevance':float(relevance[i])} for rank,i in enumerate(order)]


def region_proposals(segments,level,point=None,box=None,limit=18):
    if level not in (1,2,3):raise ValueError('无效 SAM 尺度')
    labels=segments[level];height,width=labels.shape
    if point is not None:
        x,y=map(int,point)
        if not 0<=x<width or not 0<=y<height:raise ValueError('点击位置超出照片')
        ids=[int(labels[y,x])] if labels[y,x]>=0 else []
        roi=np.zeros_like(labels,dtype=bool);roi[y,x]=True
    elif box is not None:
        x1,y1,x2,y2=map(int,box)
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:raise ValueError('框选范围无效')
        roi=np.zeros_like(labels,dtype=bool);roi[y1:y2,x1:x2]=True
        ids=[int(i) for i in np.unique(labels[roi]) if i>=0]
    else:raise ValueError('请选择点击位置或矩形范围')
    records=[]
    for index in ids:
        mask=labels==index;y,x=np.where(mask);intersection=int((mask&roi).sum())
        score=intersection/max(int((mask|roi).sum()),1) if box is not None else 1.
        records.append({'id':index,'level':level,'pixels':int(mask.sum()),'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
            'box_iou':score,'region_inside_box':intersection/max(int(mask.sum()),1)})
    return sorted(records,key=lambda r:r['box_iou'],reverse=True)[:limit]
