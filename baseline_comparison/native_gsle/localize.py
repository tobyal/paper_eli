"""Disclosed text/geometry-only instance adaptation after LangSplat querying."""
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['ROOT'])/'baseline_comparison/vendor/gsle_native'))
from common import *
from world_up import estimate_up
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw,ImageOps


def box_overlay(image,camera,corners):
    im=image.copy();d=ImageDraw.Draw(im);xy,z=camera.project(np.asarray(corners))
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if i<j and z[i]>0 and z[j]>0:d.line([tuple(xy[i]),tuple(xy[j])],fill=(255,210,0),width=3)
    return im


def main():
    cfg=config();lc=cfg['native_localization'];dest=OUT/'localization'
    source_path=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    source=GaussianData.load(source_path)
    xyz=source.xyz.cpu().numpy();scales=source.log_scale.exp().cpu().numpy();opacity=source.opacity_logit.sigmoid().cpu().numpy()[:,0]
    cameras=source_cameras(512);positions=np.array([c.center for c in cameras]);extent=float(np.median(np.linalg.norm(positions-positions.mean(0),axis=1)))
    queries=[np.load(dest/f'language_query_level{i}.npz') for i in (1,2,3)]
    floor=np.max([q['floor_scores'] for q in queries],axis=0)
    up,up_record=estimate_up(xyz,scales,opacity,floor,cameras)
    save_json(dest/'world_up.json',up_record)
    candidates=[];candidate_ids={}
    for level,q in enumerate(queries,1):
        score=q['scores'];valid=(score>lc['semantic_threshold'])&(score>q['floor_scores'])&(opacity>lc['opacity_threshold'])&np.isfinite(xyz).all(1)&(scales.max(1)<extent*.1)
        ids=np.flatnonzero(valid)
        if len(ids)<lc['minimum_component_points']:continue
        voxel=max(extent*lc['voxel_scene_fraction'],float(np.median(scales[ids].max(1)))*lc['voxel_scale_multiplier'])
        cells,inverse=np.unique(np.floor(xyz[ids]/voxel).astype(np.int64),axis=0,return_inverse=True)
        pairs=cKDTree(cells).query_pairs(1.75,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(cells),len(cells)))
        count,labels=connected_components(graph,directed=False);point_labels=labels[inverse]
        for label,n in enumerate(np.bincount(point_labels,minlength=count)):
            if n<lc['minimum_component_points'] or n>len(xyz)*.2:continue
            group=ids[point_labels==label];low,high=np.quantile(xyz[group],lc['bbox_quantiles'],axis=0)
            if np.any(high-low<=0):continue
            evidence=float(np.sum((score[group]-.5)*opacity[group])/np.sqrt(n))
            cid=f'level{level}_component{label}'
            candidates.append(dict(id=cid,level=level,points=int(n),mean_relevance=float(score[group].mean()),
                 normalized_positive_evidence=evidence,voxel_size=voxel,bbox_low=low.tolist(),bbox_high=high.tolist(),center=((low+high)/2).tolist()))
            candidate_ids[cid]=group
        print('Connected semantic candidates',level,len(ids),len(cells),len(candidates),flush=True)
    assert candidates,'No spatially supported text-positive LangSplat component; no reference-mask fallback permitted'
    candidates.sort(key=lambda x:x['normalized_positive_evidence'],reverse=True)
    chosen=candidates[0];seed=candidate_ids[chosen['id']];low=np.array(chosen['bbox_low']);high=np.array(chosen['bbox_high'])
    margin=(high-low)*lc['bbox_margin'];low-=margin;high+=margin;corners=bbox_corners(low,high);center=(low+high)/2
    inside=((xyz>=low)&(xyz<=high)).all(1)
    if 'offset' in up_record:
        inside&=~((abs(xyz@up+up_record['offset'])<up_record['tolerance'])&(floor>queries[chosen['level']-1]['scores']))
    ids=np.flatnonzero(inside);keep=np.flatnonzero(~inside)
    assert 100<len(ids)<len(xyz)*.2,('Unreasonable selected volume',len(ids))
    for name,value in [('seed_indices',seed),('target_indices',ids),('keep_indices',keep)]:np.save(dest/(name+'.npy'),value)
    np.save(OUT/'target_indices.npy',ids)
    selected=source.subset(ids);retained=source.subset(keep)
    selected.save(dest/'target.ply');retained.save(dest/'scene_without_target.ply')
    flags=torch.zeros(len(xyz),device='cuda');flags[ids]=1
    ranks=[]
    with torch.no_grad():
        for i,cam in enumerate(source_cameras(256)):
            contribution=raster(cam,source,flags[:,None].repeat(1,3))[0]
            ranks.append(dict(index=i,name=cam.name,pixels=int((contribution>.3).sum()),contribution=float(contribution.sum())))
    ranks.sort(key=lambda x:x['pixels'],reverse=True);reference=cameras[ranks[0]['index']]
    # Additional review cameras must be separated along the original capture.
    review=[ranks[0]]
    for row in ranks[1:]:
        if all(abs(row['index']-r['index'])>=10 for r in review):review.append(row)
        if len(review)==3:break
    target=dict(text=cfg['target_text'],reference_camera=reference.record(),selected_level=chosen['level'],seed_count=len(seed),target_gaussians=len(ids),retained_gaussians=len(keep),
       bbox_low=low.tolist(),bbox_high=high.tolist(),bbox_corners=corners.tolist(),center=center.tolist(),up=up.tolist(),up_estimation=up_record,
       source_ply=str(source_path),manual_roi_used=False,source_indices_preserved=True,reference_camera_source='Chosen AFTER 3D localization, maximum actual visible contribution across full COLMAP',
       extent_rule='0.5/99.5% component quantiles plus 5% margin',removal_rule='Centers in independently localized box, with supported semantic floor preservation',
       instance_selection_adapter='Voxel 26-neighbor components; maximize sum((text_relevance-0.5)*opacity)/sqrt(component_count) across three scales',
       selected_component=chosen,semantic_boundary=.5)
    save_json(dest/'target.json',target);save_json(OUT/'target_bbox.json',target);save_json(RECORDS/'localization.json',target)
    save_json(dest/'candidates.json',candidates);save_json(dest/'camera_ranking.json',ranks);save_json(dest/'review_cameras.json',[cameras[r['index']].record() for r in review])
    # Exactly the localization box is also the generation volume.
    region={**target,'object_scale':float((high-low).max()/(2*cfg['initialization']['sphere_radius'])),
            'generation_region_rule':'Identical to LangSplat-derived localization bbox; no height expansion'}
    save_json(dest/'edit_region.json',region)
    with torch.no_grad():
        for k,row in enumerate(review):
            cam=cameras[row['index']];rgb=render(cam,source,aux=False)['rgb'];mask=raster(cam,source,flags[:,None].repeat(1,3))[0]>.3
            colors=source.colors(cam.center);colors[ids]=colors[ids]*.4+colors.new_tensor([1,0,0])*.6
            save_image(dest/f'view_{k}_original.png',rgb);save_image(dest/f'view_{k}_selected.png',raster(cam,source,colors))
        save_image(OUT/'original_view.png',render(reference,source,aux=False)['rgb'])
        save_image(OUT/'localization_selected_gaussians.png',render(reference,selected,aux=False)['rgb'])
        save_image(OUT/'scene_without_target.png',render(reference,retained,aux=False)['rgb'])
        save_image(OUT/'localization_selected_mask.png',raster(reference,source,flags[:,None].repeat(1,3))[0]>.3)
        scores=torch.tensor(queries[chosen['level']-1]['scores'],device='cuda');heat=torch.stack((scores,torch.zeros_like(scores),1-scores),1)
        save_image(OUT/'language_query_heatmap.png',raster(reference,source,heat))
    box_overlay(Image.open(OUT/'original_view.png').convert('RGB'),reference,corners).save(OUT/'localization_3d_box.png')
    # Render actual alternate semantic components; no manually injected proposal.
    sheet=Image.new('RGB',(3*512,2*540),'white');draw=ImageDraw.Draw(sheet)
    with torch.no_grad():
        for k,row in enumerate(candidates[:6]):
            group=candidate_ids[row['id']];cam=min(cameras,key=lambda c:np.linalg.norm(c.center-np.asarray(row['center'])))
            # Use a calibrated camera that projects most candidate centers inside.
            sampled=xyz[group[::max(1,len(group)//300)]]
            def visibility(c):
                xy,z=c.project(sampled);return int(((z>0)&(xy[:,0]>=0)&(xy[:,0]<c.width)&(xy[:,1]>=0)&(xy[:,1]<c.height)).sum())
            cam=max(cameras,key=visibility);colors=source.colors(cam.center);colors[group]=colors[group]*.4+colors.new_tensor([1,0,0])*.6
            path=dest/f'candidate_{k}.png';save_image(path,raster(cam,source,colors))
            x,y=k%3*512,k//3*540;draw.text((x+5,y+5),f'{row["id"]} | score {row["normalized_positive_evidence"]:.3f}',fill='black')
            sheet.paste(ImageOps.contain(Image.open(path),(512,512)),(x,y+28))
    sheet.save(OUT/'localization_candidates.png');print('Selected native target',chosen,'removed',len(ids),flush=True)


if __name__=='__main__':main()
