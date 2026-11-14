"""Ten confirmed M1 targets: text-only control, immutable assisted results, audit renders.

Only columns 0/1 of the frozen language-field cache enter text-only localization.
Column 2 (reference CLIP cosine), reference mask/camera and target coordinates are
never arguments to text_select. Its component rule is the existing native GSLE
adapter, not a new official LangSplat instance-selection claim.
"""
import os, sys, json, hashlib, shutil, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F08_localization_revised'
os.environ.update(ROOT=str(ROOT),GSLE_OUTPUT=str(OUT/'adapter'),GSLE_RECORDS=str(OUT/'records'),SWAP_DISABLE_QWEN='1')
sys.path.insert(0,str(ROOT/'gsle'))
import common as c
from world_up import estimate_up

def read(p): return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def text_select(xyz,scales,opacity,cameras,fields):
    # Numerically the same component construction/ranking as native_gsle/localize.py.
    positions=np.array([cam.center for cam in cameras])
    extent=float(np.median(np.linalg.norm(positions-positions.mean(0),axis=1)))
    floor=np.max([q[:,1] for q in fields],axis=0)
    up,up_record=estimate_up(xyz,scales,opacity,floor,cameras)
    candidates=[];groups={}
    for level,q in enumerate(fields,1):
        score=q[:,0]
        valid=(score>.5)&(score>q[:,1])&(opacity>.1)&np.isfinite(xyz).all(1)&(scales.max(1)<extent*.1)
        ids=np.flatnonzero(valid)
        if len(ids)<100:continue
        voxel=max(extent*.002,float(np.median(scales[ids].max(1)))*2.)
        cells,inverse=np.unique(np.floor(xyz[ids]/voxel).astype(np.int64),axis=0,return_inverse=True)
        pairs=cKDTree(cells).query_pairs(1.75,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(cells),len(cells)))
        count,labels=connected_components(graph,directed=False);labels=labels[inverse]
        for label,n in enumerate(np.bincount(labels,minlength=count)):
            if n<100 or n>len(xyz)*.2:continue
            group=ids[labels==label];low,high=np.quantile(xyz[group],[.005,.995],axis=0)
            if np.any(high-low<=0):continue
            evidence=float(np.sum((score[group]-.5)*opacity[group])/np.sqrt(n))
            key=f'level{level}_component{label}';groups[key]=group
            candidates.append(dict(id=key,level=level,points=int(n),normalized_positive_evidence=evidence,
                mean_relevance=float(score[group].mean()),voxel_size=voxel,bbox_low=low.tolist(),bbox_high=high.tolist(),center=((low+high)/2).tolist()))
    if not candidates:raise ValueError('No text-only component; no photo fallback allowed')
    candidates.sort(key=lambda x:x['normalized_positive_evidence'],reverse=True)
    chosen=candidates[0];low=np.array(chosen['bbox_low']);high=np.array(chosen['bbox_high'])
    margin=(high-low)*.05;low-=margin;high+=margin
    inside=((xyz>=low)&(xyz<=high)).all(1)
    if 'offset' in up_record:
        inside&=~((abs(xyz@up+up_record['offset'])<up_record['tolerance'])&(floor>fields[chosen['level']-1][:,0]))
    ids=np.flatnonzero(inside)
    if not 100<len(ids)<len(xyz)*.2:raise ValueError('Unreasonable text-only volume')
    return ids,dict(selected_component=chosen,bbox_low=low.tolist(),bbox_high=high.tolist(),center=((low+high)/2).tolist(),
        up=up.tolist(),up_estimation=up_record,candidates=candidates)

def diagnostic(mask,ref):
    inter=int((mask&ref).sum())
    return dict(iou=inter/max(int((mask|ref).sum()),1),recall=inter/max(int(ref.sum()),1),precision=inter/max(int(mask.sum()),1))

@torch.no_grad()
def render_selection(scene,cam,ids,folder,name):
    flags=torch.zeros_like(scene.xyz);flags[torch.as_tensor(ids,device='cuda')]=1
    contribution=c.raster(cam,scene,flags)[0].cpu().numpy()
    mask=contribution>.05
    colors=scene.colors(cam.center);colors[ids]=.5*colors[ids]+.5*colors.new_tensor([1,0,0])
    c.save_image(folder/(name+'_highlight.png'),c.raster(cam,scene,colors))
    Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+'_mask.png'))
    return mask

def board(folder,tag,views):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    sheet=Image.new('RGB',(1800,4*368),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,paths) in enumerate(views):
        draw.text((8,i*368+4),tag+' / '+label,font=font,fill='black')
        for j,p in enumerate(paths):
            im=ImageOps.contain(Image.open(p).convert('RGB'),(594,334))
            sheet.paste(im,(j*600,i*368+30))
    sheet.save(folder/'review_sheet.jpg',quality=95)

def main():
    OUT.mkdir(parents=True,exist_ok=True);catalog=read(ROOT/'industrial_swap_v2/configs/m2_case_description_options.json')
    assert len(catalog['cases'])==10
    excluded=[x['case_id'] for x in catalog['archived_cases']]
    scene_path=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    scene=c.GaussianData.load(scene_path);xyz=scene.xyz.cpu().numpy();scales=scene.log_scale.exp().cpu().numpy();opacity=scene.opacity_logit.sigmoid().cpu().numpy()[:,0]
    cameras=c.source_cameras(512);inspection={cam.name:cam for cam in c.source_cameras(938)}
    index=[];tasks=[];shared=[];scenehash=sha(scene_path)
    for item in sorted(catalog['cases'],key=lambda x:x['case_number']):
        started=time.time();cid=item['case_id'];assert cid not in excluded
        src=ROOT/'industrial_swap_v2/webapp/data/m1_assist/cases'/cid
        conf=read(src/'confirmed_case.json');job=read(src/'job.json');meta=read(src/'semantic/semantic_provenance.json')
        assert conf['id']==cid and not conf['validation_only']
        text=job['text'];assert text==item['original_text']==meta['texts'][0]
        assert meta['columns'][:2]==['text_relevance','floor_relevance']
        assert all(q['source_parameters_identical'] for q in meta['levels'])
        tag=f"M1-{item['case_number']:02d}";d=OUT/'cases'/cid;d.mkdir(parents=True,exist_ok=True)
        photo_ids=np.load(src/'localization/target_indices.npy');target=read(src/'localization/target.json')
        fields=[np.load(src/'semantic'/f'scores_level_{i}.npy',mmap_mode='r')[:,:2] for i in (1,2,3)]
        if (d/'text_only_indices.npy').exists():ids=np.load(d/'text_only_indices.npy');selected=read(d/'text_only.json')
        else:
            ids,selected=text_select(xyz,scales,opacity,cameras,fields)
            selected.update(text=text,uses_reference_photo=False,uses_reference_mask=False,uses_reference_clip=False,
                semantic_columns=[0,1],component_algorithm='existing baseline_comparison/native_gsle/localize.py',
                gaussian_count=len(ids),source_ply=str(scene_path),source_sha256=scenehash)
            np.save(d/'text_only_indices.npy',ids);save(d/'text_only.json',selected)
        shutil.copy2(src/'localization/target_indices.npy',d/'photo_assisted_indices.npy')
        shutil.copy2(src/'reference/reference_mask.png',d/'reference_mask.png')
        shutil.copy2(src/'reference/original_photo.jpg',d/'reference_photo.jpg')
        shutil.copy2(src/'reference/selection_overlay.jpg',d/'reference_selection.jpg')
        result=read(src/'result.json');ref=c.Camera.from_record(result['views'][0]['camera'])
        views=[ref]+[c.Camera.from_record(x['camera']) for x in result['views'][1:3]]
        assert len(views)==3 and all(v.name!=ref.name for v in views[1:])
        masks={};viewrecords=[];rows=[]
        for k,cam in enumerate(views):
            vd=d/f'view_{k}';vd.mkdir(exist_ok=True)
            original=ROOT/'data/industrial/images'/cam.name
            Image.open(original).convert('RGB').resize((cam.width,cam.height),Image.Resampling.LANCZOS).save(vd/'photograph.png')
            with torch.no_grad():c.save_image(vd/'scene.png',c.render(cam,scene,aux=False)['rgb'])
            m={method:render_selection(scene,cam,ix,vd,method) for method,ix in [('text_only',ids),('photo_assisted',photo_ids)]}
            save(vd/'camera.json',cam.record());masks[k]=m
            viewrecords.append(dict(camera=cam.record(),role='input_reference' if k==0 else 'held_out_from_instance_selection',
                text_visible_pixels=int(m['text_only'].sum()),photo_visible_pixels=int(m['photo_assisted'].sum())))
            if k:
                task_id=f'{tag}_{cam.name.split(".")[0]}';ad=OUT/'annotation_tasks'/task_id;ad.mkdir(parents=True,exist_ok=True)
                shutil.copy2(vd/'photograph.png',ad/'image.png');shutil.copy2(d/'reference_selection.jpg',ad/'target_identity_reference.jpg')
                save(ad/'camera.json',cam.record())
                tasks.append(dict(task_id=task_id,m1_case_id=cid,label=tag,camera=cam.record(),input_camera=ref.name,
                    original_photo=str(original),image_sha256=sha(ad/'image.png'),
                    task_image=str(ad/'image.png'),target_identity_reference=str(ad/'target_identity_reference.jpg'),
                    mask_path=str(ad/'mask.png'),review_path=str(ad/'review.json'),annotation_status='pending_independent_human_annotation',
                    previously_available_m1_inspection_view=True,used_for_instance_selection=False,
                    pretraining_note='Original scene and language field were trained using the registered capture; held out only from instance-specific SAM assistance.',
                    predicted_masks={meth:str(vd/(meth+'_mask.png')) for meth in m}))
            rows.append((('input reference' if k==0 else f'held-out view {k}')+' | photograph / text-only / photo-assisted',
                [d/'reference_selection.jpg' if k==0 else vd/'photograph.png',vd/'text_only_highlight.png',vd/'photo_assisted_highlight.png']))
        # Show actual text-selected instance from its own calibrated observation, after selection.
        points=xyz[ids[::max(1,len(ids)//700)]]
        def visible(cam):
            uv,z=cam.project(points);return int(((z>.01)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)).sum())
        best_candidates=sorted(cameras,key=visible,reverse=True)[:16]
        flag=torch.zeros_like(scene.xyz);flag[ids]=1
        with torch.no_grad():native=max(best_candidates,key=lambda cam:float(c.raster(cam,scene,flag)[0].sum()))
        native=inspection[native.name];nv=d/'native_text';nv.mkdir(exist_ok=True)
        Image.open(ROOT/'data/industrial/images'/native.name).resize((native.width,native.height),Image.Resampling.LANCZOS).save(nv/'photograph.png')
        render_selection(scene,native,ids,nv,'text_only');render_selection(scene,native,photo_ids,nv,'photo_assisted');save(nv/'camera.json',native.record())
        rows.append(('text-only own inspection view | photograph / text-only / intended instance',[nv/'photograph.png',nv/'text_only_highlight.png',nv/'photo_assisted_highlight.png']))
        board(d,tag,rows)
        refmask=np.array(Image.open(d/'reference_mask.png').resize((ref.width,ref.height),Image.Resampling.NEAREST))>127
        both=np.intersect1d(ids,photo_ids)
        diag={method:diagnostic(mask,refmask) for method,mask in masks[0].items()}
        delta=float(np.linalg.norm(np.array(selected['center'])-np.array(target['center'])))
        row=dict(m1_case_id=cid,label=tag,catalog_number=item['case_number'],description=text,source_case=str(src),
            user_confirmation=conf,reference_camera=ref.record(),reference_sam=str(src/'reference/reference_mask.png'),
            semantic_provenance=meta,semantic_files=[dict(path=str(src/'semantic'/f'scores_level_{i}.npy'),sha256=sha(src/'semantic'/f'scores_level_{i}.npy')) for i in (1,2,3)],
            photo_indices_source=str(src/'localization/target_indices.npy'),photo_indices_sha256=sha(src/'localization/target_indices.npy'),
            text_indices_sha256=sha(d/'text_only_indices.npy'),text_count=len(ids),photo_count=len(photo_ids),
            text_center=selected['center'],photo_center=target['center'],center_distance=delta,
            shared_gaussian_count=len(both),shared_gaussian_iou=len(both)/len(np.union1d(ids,photo_ids)),
            input_sam_diagnostic=diag,views=viewrecords,text_native_camera=native.record(),review_sheet=str(d/'review_sheet.jpg'),
            seconds=time.time()-started)
        index.append(row);save(d/'evaluation.json',row);save(OUT/'case_index.json',dict(cases=index,excluded=excluded,catalog=str(ROOT/'industrial_swap_v2/configs/m2_case_description_options.json')))
        save(OUT/'evaluation_views.json',dict(views=tasks,annotation_count=0,annotation_policy='Independent human-reviewed visible masks only; no prediction or input-SAM-derived ground truth.'))
        print(tag,cid,'text',len(ids),'photo',len(photo_ids),'overlap',row['shared_gaussian_iou'],'sam',diag,'seconds',round(time.time()-started),flush=True)
    save(OUT/'sources.json',dict(scene=dict(path=str(scene_path),sha256=scenehash),
        formal_m1_algorithm_unchanged=True,executed_stages=['M1 text-only control','Gaussian inspection rendering'],
        annotation_search_roots=['data','records','industrial_swap_v2/webapp/data/m1_assist','industrial_swap_v2/outputs/paper_figures_v4'],
        independent_masks_found=0,input_sam_not_ground_truth=True,
        scripts={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'baseline_comparison/native_gsle/localize.py',ROOT/'industrial_swap_v2/scripts/m1_language.py']},
        models=index[0]['semantic_provenance'],case_index='case_index.json',evaluation_views='evaluation_views.json'))
    print('READY FOR INSTANCE REVIEW; independent annotations pending',flush=True)

if __name__=='__main__':main()
