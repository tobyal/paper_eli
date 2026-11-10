"""Registered-camera, instance-aware qualitative figures; no scene edits/training.

Camera ranking uses actual Gaussian visibility, framing and source-photo
reprojection quality. Appearance/semantic success of the final edit is not a
ranking criterion. All localization indices are retained in the displayed mask.
"""
import argparse
import html
import math
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import render as r
import localization_red as lr
from target_centered_pairs import own_reference

OUT=r.BASE/'viewpoint_aligned_v2'
METHODS=lr.METHODS
LABELS=lr.LABELS
CASES=['case00','case04','case02','case07','case10']
INPUT_NOTE='Native baseline localization; Ours additionally uses a reference photograph for target-instance specification.'
RULE='Identical cameras within each column; registered camera sharing when the selected instance and visibility permit.'


def bbox(points):
    lo,hi=np.quantile(points,[.005,.995],axis=0)
    return dict(low=lo.tolist(),high=hi.tolist(),center=((lo+hi)/2).tolist(),diagonal=float(np.linalg.norm(hi-lo)))


@torch.no_grad()
def focus_component(ids, original, native, source_xyz, method, case):
    if method=='dge':
        old=r.BASE/'matched_views'/case
        focus=np.load(old/'dge_display_focus_indices.npy')
        assert np.isin(focus,ids).all()
        notes=r.read(old/'camera_params.json')['methods']['dge']['target']['component_framing']
        return focus,dict(policy='Reuse independently computed native-primary-visible framing component; ALL indices remain in red visualization.',previous_evidence=notes)
    pts=source_xyz[ids]
    radius=float(np.linalg.norm(np.diff(np.quantile(source_xyz,[.005,.995],axis=0),axis=0))*.003)
    dist,nn=cKDTree(pts).query(pts,k=min(32,len(pts)),distance_upper_bound=radius)
    rr=np.broadcast_to(np.arange(len(pts))[:,None],nn.shape);valid=nn<len(pts)
    graph=coo_matrix((np.ones(valid.sum()),(rr[valid],nn[valid])),shape=(len(pts),len(pts)))
    count,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    top=np.argsort(sizes)[-8:][::-1];scores=[]
    cam=native.resize(320,round(320*native.height/native.width))
    for start in range(0,len(top),3):
        group=top[start:start+3];flags=torch.zeros_like(original.xyz)
        for j,k in enumerate(group):flags[torch.as_tensor(ids[labels==k],device='cuda'),j]=1
        mass=r.c.raster(cam,original,flags).sum((1,2)).cpu().numpy()
        for j,k in enumerate(group):scores.append(dict(component=int(k),gaussians=int(sizes[k]),native_visible_mass=float(mass[j]),bounds=bbox(pts[labels==k])))
    best=max(scores,key=lambda x:x['native_visible_mass']);focus=ids[labels==best['component']]
    return focus,dict(policy='Primary framing component by own native camera contribution, never by Ours overlap.',
        component_count=count,radius=radius,selected=best,top_components=scores,
        primary_fraction=len(focus)/len(ids),full_selection_preserved=True)


def instance_relation(a,b):
    full=int(np.intersect1d(a['ids'],b['ids']).size)
    shared=int(np.intersect1d(a['focus'],b['focus']).size)
    overlap=shared/min(len(a['focus']),len(b['focus']))
    ba,bb=a['local_bounds'],b['local_bounds']
    ca,cb=np.asarray(ba['center']),np.asarray(bb['center'])
    distance=float(np.linalg.norm(ca-cb));relative=distance/max(ba['diagonal'],bb['diagonal'],1e-6)
    separation=np.maximum(0,np.maximum(np.asarray(ba['low'])-bb['high'],np.asarray(bb['low'])-ba['high']))
    gap=float(np.linalg.norm(separation))
    if overlap>=.55 and relative<.8:
        relation='same_instance';reason='Strong shared source-Gaussian identity in the independently selected primary region, with compatible 3D centers/extents.'
    elif overlap<=.02 and (gap>.03*max(ba['diagonal'],bb['diagonal']) or relative>.65):
        relation='different_instance';reason='Almost disjoint primary Gaussian sets and spatially separated 3D regions.'
    else:
        relation='uncertain';reason='Partial/mixed selection or insufficient geometric separation; not forced into a shared-instance group.'
    return dict(relation=relation,reason=reason,full_shared_indices=full,
        full_overlap_over_smaller=full/min(len(a['ids']),len(b['ids'])),
        primary_shared_indices=shared,primary_overlap_over_smaller=overlap,
        primary_center_distance=distance,relative_center_distance=relative,bbox_gap=gap,
        a_primary_count=len(a['focus']),b_primary_count=len(b['focus']),
        note='A same-instance primary region does not imply identical segmentation. Disconnected selections and leakage are retained and separately recorded.')


def angles(cam,center,up):
    ori=r.orientation(cam.center-center,up);u=cam.R@up
    ori.update(roll=float(np.degrees(np.arctan2(u[0],-u[1]))),
               fov_x=float(np.degrees(2*np.arctan(cam.width/(2*cam.fx)))),
               fov_y=float(np.degrees(2*np.arctan(cam.height/(2*cam.fy)))))
    return ori


def difference(cam,center,ref,refcenter,up):
    a=angles(cam,center,up);b=angles(ref,refcenter,up)
    return dict(yaw=r.delta_angle(a['yaw'],b['yaw']),pitch=a['pitch']-b['pitch'],
        roll=r.delta_angle(a['roll'],b['roll']),fov_x=a['fov_x']-b['fov_x'],fov_y=a['fov_y']-b['fov_y'])


def mask_stats(alpha):
    result=r.alpha_stats(alpha);h,w=alpha.shape;box=result['bbox']
    if box:
        result['size']=max((box[2]-box[0])/w,(box[3]-box[1])/h)
        result['center_normalized']=[(box[0]+box[2])/(2*w),(box[1]+box[3])/(2*h)]
    else:result['center_normalized']=[-2,-2]
    return result


def project_score(cam,old_bounds,new_bounds,center,up):
    score=0.;boxes=[]
    for bounds in [old_bounds,new_bounds]:
        xy,z=cam.project(r.c.bbox_corners(*bounds))
        if (z<=.04).any():return 1e6
        lo,hi=xy.min(0)/[cam.width,cam.height],xy.max(0)/[cam.width,cam.height]
        size=np.max(hi-lo);area=np.prod(np.maximum(hi-lo,1e-6))
        clipped=np.prod(np.maximum(0,np.minimum(hi,1)-np.maximum(lo,0)))/area
        score+=12*(1-clipped)+abs(math.log(max(size,.001)/.60))
        boxes.append((lo+hi)/2)
    midpoint=np.mean(boxes,axis=0);score+=2*np.linalg.norm(midpoint-[.5,.55])
    pitch=angles(cam,center,up)['pitch'];score+=.015*max(0,12-pitch)+.015*max(0,pitch-50)
    return float(score)


def shortlist(ctx,cams,up):
    old=[ctx['local_bounds']['low'],ctx['local_bounds']['high']]
    scores=[project_score(cam,old,ctx['new_bounds'],ctx['center'],up) for cam in cams]
    ordered=np.argsort(scores).tolist();chosen=ordered[:18]
    # Keep alternate trajectory directions available for reference matching.
    sectors={}
    for i in ordered:
        if scores[i]>=12:continue
        sector=int((angles(cams[i],ctx['center'],up)['yaw']+180)//45)%8
        if sector not in sectors:sectors[sector]=i;chosen.append(i)
    name=ctx['native'].name.replace('.jpg','')
    chosen += [i for i,c in enumerate(cams) if c.name.replace('.jpg','')==name]
    return list(dict.fromkeys(chosen)),scores


class Evaluator:
    def __init__(self,original,cams,up):
        self.original=original;self.cams=cams;self.up=up;self.source_cache={}

    @torch.no_grad()
    def source(self,i):
        if i not in self.source_cache:
            cam=self.cams[i];low=cam.resize(320,round(320*cam.height/cam.width))
            rgb=r.c.render(low,self.original,aux=False)['rgb'].cpu().numpy().transpose(1,2,0)
            photo=r.ROOT/'data/industrial/images'/cam.name
            actual=np.asarray(Image.open(photo).convert('RGB').resize((low.width,low.height),Image.Resampling.LANCZOS))/255.
            self.source_cache[i]=(rgb,actual)
        return self.source_cache[i]

    @torch.no_grad()
    def evaluate(self,ctx,i):
        if i in ctx['metrics']:return ctx['metrics'][i]
        cam=self.cams[i];low=cam.resize(320,round(320*cam.height/cam.width));w,h=low.width,low.height
        guard=r.c.Camera(low.R,low.t,w*2,h*2,low.fx,low.fy,low.cx+w/2,low.cy+h/2,'expanded_frustum')
        old=r.object_alpha(low,ctx['old']);obj=r.object_alpha(low,ctx['obj'])
        old_guard=r.object_alpha(guard,ctx['old']);obj_guard=r.object_alpha(guard,ctx['obj'])
        src=r.c.raster(low,self.original,ctx['source_flags']).cpu().numpy()
        vis=r.c.raster(low,ctx['vis_scene'],ctx['flags']).cpu().numpy()
        os,ns=mask_stats(old),mask_stats(obj)
        capture_old=float(old.sum()/max(old_guard.sum(),1e-8));capture_new=float(obj.sum()/max(obj_guard.sum(),1e-8))
        visibility_old=float(src[0].sum()/max(old.sum(),1e-8));visibility_new=float(vis[0].sum()/max(obj.sum(),1e-8))
        rgb,photo=self.source(i);environment=(src[1]<.03)&(obj<.03)&(src[2]>.1)
        bg_mae=float(np.abs(rgb-photo)[environment].mean()) if environment.sum()>20 else 1.
        coverage=float((src[2]>.1).mean())
        center=.5*(np.array(os['center_normalized'])+np.array(ns['center_normalized']))
        pitch=angles(cam,ctx['center'],self.up)['pitch']
        score=35*max(0,.98-capture_old)+40*max(0,.98-capture_new)
        score+=5*max(0,.65-visibility_old)+7*max(0,.78-visibility_new)
        score+=.7*abs(math.log(max(os['size'],.001)/.60))+abs(math.log(max(ns['size'],.001)/.60))
        score+=1.5*np.linalg.norm(center-[.5,.55])+5*bg_mae+2*max(0,.9-coverage)
        score+=.015*max(0,12-pitch)+.015*max(0,pitch-50)
        record=dict(camera_index=i,name=cam.name,score=float(score),old_capture=capture_old,new_capture=capture_new,
            old_visibility=visibility_old,new_visibility=visibility_new,old_projection=os,new_projection=ns,
            source_background_photo_mae=bg_mae,scene_coverage=coverage,
            old_visible_pixels=int((src[0]>.05).sum()),pitch=pitch,
            final_visibility_proxy=(ctx['method']=='dge'))
        ctx['metrics'][i]=record
        return record


def readable(m):
    return (m['old_capture']>=.95 and m['new_capture']>=.95 and m['old_visibility']>=.25 and
            m['new_visibility']>=.35 and .12<=m['new_projection']['size']<=.95 and m['old_visible_pixels']>=35)


def choose_group(contexts,cams,up,relations,pairwise,evaluator):
    # Evaluate the best Ours candidates also for other methods selecting the
    # same instance, to compare one identical physical camera fairly.
    ours=contexts['ours'];ours_rank=sorted(ours['metrics'],key=lambda i:ours['metrics'][i]['score'])
    refs=ours_rank[:14]
    for method,ctx in contexts.items():
        if method=='ours':continue
        if relations[method]['relation']=='same_instance':
            for i in refs:evaluator.evaluate(ctx,i)
    shared_baseline_pairs=[]
    for ia,a in enumerate(METHODS[:-1]):
        for b in METHODS[ia+1:-1]:
            if pairwise[a+'__'+b]['relation']!='same_instance':continue
            shared_baseline_pairs.append((a,b))
            common=set()
            for m in (a,b):
                common.update(sorted(contexts[m]['metrics'],key=lambda i:contexts[m]['metrics'][i]['score'])[:6])
            for m in (a,b):
                for i in common:evaluator.evaluate(contexts[m],i)
    options=[]
    for ref in refs:
        reference=ours['metrics'][ref];chosen={'ours':dict(index=ref,reason='Jointly selected registered reference camera',match_penalty=0.)}
        total=.8*reference['score']
        for method,ctx in contexts.items():
            if method=='ours':continue
            pool=[i for i,m in ctx['metrics'].items() if readable(m)]
            fallback=False
            if not pool:pool=list(ctx['metrics']);fallback=True
            own_best=min(ctx['metrics'][i]['score'] for i in pool)
            same=relations[method]['relation']=='same_instance'
            # Share an exact world camera when it remains practically readable.
            exact=ctx['metrics'].get(ref)
            if same and exact and readable(exact) and exact['score']<=own_best+1.8:
                selected=ref;penalty=0.;reason='Same-instance evidence; exact shared registered camera is readable'
            else:
                ranked=[]
                for i in pool:
                    d=difference(cams[i],ctx['center'],cams[ref],ours['center'],up)
                    size=abs(math.log(max(ctx['metrics'][i]['new_projection']['size'],.001)/max(reference['new_projection']['size'],.001)))
                    penalty=.018*abs(d['yaw'])+.035*abs(d['pitch'])+.02*abs(d['roll'])+.3*size
                    ranked.append((ctx['metrics'][i]['score']+penalty,i,penalty))
                _,selected,penalty=min(ranked)
                if same:reason='Shared camera has insufficient framing/visibility or substantially poorer source quality; alternate registered camera'
                else:reason='Independent instance: closest practical relative direction among readable registered cameras'
                if fallback:reason+='; no candidate passes all readability bounds, best available real view retained'
            chosen[method]=dict(index=selected,reason=reason,match_penalty=penalty,
                same_world_camera=selected==ref,relaxed_readability=fallback)
            total+=(ctx['metrics'][selected]['score']+.6*penalty)/3
            if same and selected!=ref:total+=.8
        # Same-instance sharing also applies to two baselines that selected an
        # instance different from Ours, not only to comparisons against Ours.
        for a,b in shared_baseline_pairs:
            if chosen[a]['index']==chosen[b]['index']:continue
            common=set(contexts[a]['metrics'])&set(contexts[b]['metrics'])
            if (relations[a]['relation']=='same_instance' or relations[b]['relation']=='same_instance'):
                common={ref}&common
            candidates=[]
            for i in common:
                if not all(readable(contexts[m]['metrics'][i]) for m in (a,b)):continue
                if any(contexts[m]['metrics'][i]['score']>min(x['score'] for x in contexts[m]['metrics'].values())+1.8 for m in (a,b)):continue
                cost=sum(contexts[m]['metrics'][i]['score'] for m in (a,b))
                candidates.append((cost,i))
            current=sum(contexts[m]['metrics'][chosen[m]['index']]['score'] for m in (a,b))
            if candidates:
                cost,i=min(candidates)
                if cost<=current+1.4:
                    for m,other in [(a,b),(b,a)]:
                        d=difference(cams[i],contexts[m]['center'],cams[ref],ours['center'],up)
                        penalty=.018*abs(d['yaw'])+.035*abs(d['pitch'])+.02*abs(d['roll'])
                        chosen[m]=dict(index=i,reason='Same-instance baseline pair shares an identical readable registered camera with '+other,
                            match_penalty=penalty,same_world_camera=i==ref,relaxed_readability=False)
        total=.8*reference['score']
        for method in METHODS[:-1]:
            selected=chosen[method];total+=(contexts[method]['metrics'][selected['index']]['score']+.6*selected['match_penalty'])/3
            if relations[method]['relation']=='same_instance' and selected['index']!=ref:total+=.8
        total+=.5*sum(chosen[a]['index']!=chosen[b]['index'] for a,b in shared_baseline_pairs)
        options.append(dict(reference_index=ref,joint_score=float(total),assignments=chosen))
    best=min(options,key=lambda x:x['joint_score'])
    return best,options


def overlay(rgb,contribution):
    rgb=np.clip(rgb,0,1);weight=.25*np.clip(contribution,0,1)[...,None]
    tinted=rgb*(1-weight)+np.array([1.,0.,0.])*weight
    image=np.uint8(np.clip(tinted*255,0,255))
    mask=np.uint8(contribution>.15)*255
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # Overlay includes every pixel; suppress only subpixel-scale contour noise.
    contours=[c for c in contours if cv2.contourArea(c)>=4]
    cv2.drawContours(image,contours,-1,(255,45,35),max(1,round(rgb.shape[1]/800)),cv2.LINE_AA)
    return Image.fromarray(image)


def shared_crop(boxes,width,height):
    valid=[b for b in boxes if b]
    if not valid:return [0,0,width,height]
    b=np.array(valid);lo=b[:,:2].min(0);hi=b[:,2:].max(0);mid=(lo+hi)/2
    # Preserve both old and new object extent; crop is identical in before/after.
    extent=(hi-lo)*1.14;cw=max(extent[0],extent[1]*4/3);ch=cw*3/4
    scale=min(1,width/cw,height/ch);cw*=scale;ch*=scale
    x=np.clip(mid[0]-cw/2,0,width-cw);y=np.clip(mid[1]-ch/2,0,height-ch)
    return [int(x),int(y),int(math.ceil(x+cw)),int(math.ceil(y+ch))]


def board(path,rows,title,subtitle,cell=800,ratio=902/1600):
    ch=round(cell*ratio);header=100;rowheight=ch+45;foot=52
    canvas=Image.new('RGB',(4*cell,header+len(rows)*rowheight+foot),'white');d=ImageDraw.Draw(canvas)
    d.text((18,12),title,font=ImageFont.truetype('DejaVuSans.ttf',32),fill='black')
    d.text((18,58),subtitle,font=ImageFont.truetype('DejaVuSans.ttf',24),fill='#333333')
    for rownum,row in enumerate(rows):
        for col,(label,file) in enumerate(row):
            x,y=col*cell,header+rownum*rowheight
            d.text((x+12,y+6),label,font=ImageFont.truetype('DejaVuSans.ttf',29),fill='black')
            im=Image.open(file).convert('RGB');im=ImageOps.contain(im,(cell,ch),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+(cell-im.width)//2,y+45+(ch-im.height)//2))
    d.text((18,canvas.height-37),INPUT_NOTE,font=ImageFont.truetype('DejaVuSans.ttf',23),fill='#333333')
    canvas.save(path,dpi=(300,300))


@torch.no_grad()
def run(case):
    started=time.time();out=OUT/case;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'camera_matching.json').exists(),'Completed result is preserved; use a different version directory'
    spec=r.read(r.BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']==case)
    before={spec['source_scene']:r.sha(spec['source_scene'])}
    original=r.c.GaussianData.load(spec['source_scene']);source_xyz=original.xyz.cpu().numpy()
    cams=r.c.source_cameras(1600);target=r.read(row['target_record']);up=r.c.normalized(np.array(target['up']))
    previous=r.read(r.BASE/'matched_views'/case/'camera_params.json')
    contexts={}
    for method in METHODS:
        info=dict(previous['methods'][method]['target']);ids=np.load(lr.index_path(row,method))
        native,native_source=own_reference(row,method)
        if method in ['gaussianeditor','dge']:
            focus,focus_notes=focus_component(ids,original,native,source_xyz,method,case)
            lb=bbox(source_xyz[focus]);center=np.array(lb['center'])
        else:
            focus=ids;focus_notes=dict(policy='Full saved LangSplat/M1 selected region; own localization center retained')
            lb=bbox(source_xyz[ids]);center=np.array(info['center'])
        old=original.subset(torch.as_tensor(focus,device='cuda'))
        if method=='dge':info['indices']=focus.tolist()
        scene=r.c.GaussianData.load(info['scene_path']);obj,vis_scene,flags=r.flag_data(info,scene,original)
        source_flags=torch.zeros_like(original.xyz);source_flags[torch.as_tensor(focus,device='cuda'),0]=1
        source_flags[torch.as_tensor(ids,device='cuda'),1]=1;source_flags[:,2]=1
        newpoints=obj.xyz.cpu().numpy();new_bounds=[np.quantile(newpoints,.005,axis=0).tolist(),np.quantile(newpoints,.995,axis=0).tolist()]
        if method!='dge':new_bounds=info['geometry']['bounds']
        paths=[lr.index_path(row,method),Path(info['scene_path']),Path(native_source['path'])]
        if info.get('object_path'):paths.append(Path(info['object_path']))
        for p in paths:before[str(p)]=r.sha(p)
        contexts[method]=dict(method=method,ids=ids,focus=focus,focus_notes=focus_notes,local_bounds=lb,
            full_bounds=bbox(source_xyz[ids]),center=center,old=old,scene=scene,obj=obj,vis_scene=vis_scene,
            flags=flags,source_flags=source_flags,new_bounds=new_bounds,native=native,native_source=native_source,
            info=info,metrics={})
        print(case,method,'target prepared',len(ids),len(focus),flush=True)
    relations={m:instance_relation(ctx,contexts['ours']) for m,ctx in contexts.items() if m!='ours'}
    pairs={a+'__'+b:instance_relation(contexts[a],contexts[b]) for i,a in enumerate(METHODS) for b in METHODS[i+1:]}
    instance=dict(case=case,against_ours=relations,pairwise=pairs,methods={})
    for m,ctx in contexts.items():
        instance['methods'][m]=dict(indices_source=str(lr.index_path(row,m)),full_count=len(ctx['ids']),primary_count=len(ctx['focus']),
            primary_center=ctx['center'].tolist(),primary_bounds=ctx['local_bounds'],full_bounds=ctx['full_bounds'],
            framing_component=ctx['focus_notes'],generated_bounds=ctx['new_bounds'],
            generated_median=np.median(ctx['obj'].xyz.cpu().numpy(),axis=0).tolist(),
            delete_to_generated_median_distance=float(np.linalg.norm(np.median(ctx['obj'].xyz.cpu().numpy(),axis=0)-ctx['center'])),
            placement=ctx['info'].get('placement'),all_indices_used_for_localization=True)
    r.save(out/'instance_matching.json',instance)
    print(case,'relations',{m:d['relation'] for m,d in relations.items()},flush=True)
    evaluator=Evaluator(original,cams,up)
    for m,ctx in contexts.items():
        indices,geom=shortlist(ctx,cams,up);ctx['geometric_scores']=geom
        for i in indices:evaluator.evaluate(ctx,i)
        print(case,m,'evaluated',len(ctx['metrics']),'registered cameras',flush=True)
    best,options=choose_group(contexts,cams,up,relations,pairs,evaluator)
    ref=best['reference_index'];refcam=cams[ref]
    audit=dict(case=case,reference_camera=refcam.record(),reference_camera_index=ref,
        source='Original registered COLMAP camera poses and calibration, scaled to the common export resolution',
        colmap_intrinsics=str(r.ROOT/'data/industrial/sparse/0/cameras.bin'),
        colmap_extrinsics=str(r.ROOT/'data/industrial/sparse/0/images.bin'),
        registered_camera_count=len(cams),virtual_adjustments=0,selection_options=options,methods={},
        input_difference=INPUT_NOTE,camera_policy=RULE,ranking_excludes_final_semantic_or_appearance_quality=True)
    for folder in ['localization','final','original','before_after','camera_candidates']:(out/folder).mkdir(exist_ok=True)
    for m,ctx in contexts.items():
        assignment=best['assignments'][m];i=assignment['index'];cam=cams[i]
        metric=evaluator.evaluate(ctx,i)
        base=r.c.render(cam,original,aux=False)['rgb'];final=r.c.render(cam,ctx['scene'],aux=False)['rgb']
        packed=r.c.raster(cam,original,ctx['source_flags']);allmask=packed[1].cpu().numpy()
        rgb=base.cpu().numpy().transpose(1,2,0);over=overlay(rgb,allmask)
        r.c.save_image(out/'original'/(m+'.png'),base);r.c.save_image(out/'final'/(m+'.png'),final)
        over.save(out/'localization'/(m+'.png'),dpi=(300,300))
        r.c.save_image(out/'localization'/(m+'_visible_contribution.png'),packed[1:2])
        old_alpha=r.object_alpha(cam,ctx['old']);new_alpha=r.object_alpha(cam,ctx['obj'])
        crop=shared_crop([mask_stats(old_alpha)['bbox'],mask_stats(new_alpha)['bbox']],cam.width,cam.height)
        for stage,source in [('before','original'),('after','final')]:
            Image.open(out/source/(m+'.png')).crop(crop).save(out/'before_after'/(m+'_'+stage+'.png'),dpi=(300,300))
        d=difference(cam,ctx['center'],refcam,contexts['ours']['center'],up)
        relaxed=abs(d['yaw'])>35 or abs(d['pitch'])>15
        audit['methods'][m]=dict(camera=cam.record(),localization_camera=cam.record(),final_camera=cam.record(),
            before_camera=cam.record(),after_camera=cam.record(),camera_index=i,registered=True,
            same_world_camera_as_ours=i==ref,instance_relation=('self' if m=='ours' else relations[m]['relation']),
            assignment=assignment,relative_view_difference=d,metrics=metric,shared_before_after_crop=crop,
            angle_constraints_relaxed=relaxed,relaxation_reason=('No closer candidate with comparably readable target/framing/source background under the ranking rule; actual registered view retained.' if relaxed else None),
            localization_style=dict(opacity=.25,mask='Actual visible contribution of ALL saved selected Gaussians',contour_threshold=.15),
            final_scene=ctx['info']['scene_path'],selected_indices=str(lr.index_path(row,m)))
        ranked=sorted(ctx['metrics'],key=lambda j:ctx['metrics'][j]['score'])
        r.save(out/'camera_candidates'/(m+'_metrics.json'),dict(geometric_scores=ctx['geometric_scores'],
            evaluated=[ctx['metrics'][j] for j in ranked],selected_index=i))
        candidates=list(dict.fromkeys([i]+ranked))[:3];panels=[]
        for j in candidates:
            candidate=cams[j].resize(640,round(640*cam.height/cam.width))
            candidate_base=r.c.render(candidate,original,aux=False)['rgb'].cpu().numpy().transpose(1,2,0)
            candidate_mask=r.c.raster(candidate,original,ctx['source_flags'])[1].cpu().numpy()
            file=out/'camera_candidates'/f'{m}_{j:03d}_localization.png';overlay(candidate_base,candidate_mask).save(file)
            f=out/'camera_candidates'/f'{m}_{j:03d}_final.png';r.c.save_image(f,r.c.render(candidate,ctx['scene'],aux=False)['rgb'])
            panels.append(dict(index=j,camera=cams[j].record(),selected=j==i,localization=file.name,final=f.name))
        r.save(out/'camera_candidates'/(m+'_shortlist.json'),panels)
        print(case,m,'FINAL',cam.name,'shared',i==ref,'yaw',round(d['yaw'],1),'visibility',round(metric['new_visibility'],3),flush=True)
    assert all(r.sha(p)==h for p,h in before.items())
    audit.update(input_sha256=before,source_files_unchanged=True,training_steps=0,ply_writes=0,seconds=time.time()-started)
    r.save(out/'camera_matching.json',audit)
    board(out/'figure_localization_final.png',[
        [(label+' / Localization',out/'localization'/(m+'.png')) for m,label in zip(METHODS,LABELS)],
        [(label+' / Final',out/'final'/(m+'.png')) for m,label in zip(METHODS,LABELS)]],
        case+' | Localization and Final Editing',RULE)
    board(out/'figure_before_after.png',[
        [(label+' / Original',out/'before_after'/(m+'_before.png')) for m,label in zip(METHODS,LABELS)],
        [(label+' / Final',out/'before_after'/(m+'_after.png')) for m,label in zip(METHODS,LABELS)]],
        case+' | Before / After Replacement Quality','Identical camera and crop within each method; full-frame selection differences remain in the main figure.',ratio=.75)
    review_page(out,case,audit,instance)
    print('COMPLETE',case,round(time.time()-started,1),flush=True)


def review_page(out,case,audit,instance):
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{case} SCI 视角与实例对齐</title><style>body{{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:10px}}section{{margin:28px 0}}</style><h1>{case}：真实 COLMAP 相机的实例感知对比</h1><p><a href="../index.html">总览</a> | <a href="camera_matching.json">相机完整记录</a> | <a href="instance_matching.json">实例关系证据</a></p><p>每列上下使用完全相同的相机。红色覆盖强度约 25%，轮廓来自真实高斯可见贡献；保留全部定位索引。最终结果来自未修改的最终 PLY，GSLE 为原始 Native DI-SDS。</p><p>输入差异：基线使用各自原生定位；Ours 额外使用一张原始场景参考照片指定实例。因此不是完全相同输入信息的定位比较。</p><table><tr><th>方法</th><th>相对 Ours 实例</th><th>COLMAP 相机</th><th>共享世界相机</th><th>Yaw / Pitch 差异</th></tr>'''
    for m,l in zip(METHODS,LABELS):
        a=audit['methods'][m];d=a['relative_view_difference']
        text+=f'<tr><td>{l}</td><td>{a["instance_relation"]}</td><td>{a["camera"]["name"]}</td><td>{a["same_world_camera_as_ours"]}</td><td>{d["yaw"]:.1f}° / {d["pitch"]:.1f}°</td></tr>'
    text+='</table>'
    for name,label in [('figure_localization_final','论文主图：定位与最终结果'),('figure_before_after','替换质量：相同区域 Before / After')]:
        text+=f'<section><h2>{label}</h2><a href="{name}.png"><img src="{name}.png"></a></section>'
    text+='<h2>主要候选视角</h2><p>以下展示实际评估的注册相机；不通过更改场景或定位 Mask 改善显示。</p>'
    for m,l in zip(METHODS,LABELS):
        text+=f'<h3>{l}</h3><p><a href="camera_candidates/{m}_metrics.json">全部候选指标</a></p><div style="display:flex;gap:12px">'
        for c in r.read(out/'camera_candidates'/(m+'_shortlist.json')):
            text+=f'<div style="width:33%"><p>{c["camera"]["name"]} {"（选中）" if c["selected"] else ""}</p><img src="camera_candidates/{c["localization"]}"><img src="camera_candidates/{c["final"]}"></div>'
        text+='</div>'
    (out/'index.html').write_text(text+'</html>',encoding='utf-8')


def package(cases):
    display=[c for c in lr.CASES if c in cases]
    images=[Image.open(OUT/c/'figure_localization_final.png').convert('RGB') for c in display]
    width=2400;resized=[im.resize((width,round(im.height*width/im.width)),Image.Resampling.LANCZOS) for im in images]
    canvas=Image.new('RGB',(width,sum(im.height for im in resized)),'white');y=0
    for im in resized:canvas.paste(im,(0,y));y+=im.height
    canvas.save(OUT/'figure_all_cases.png',dpi=(300,300))
    ImageOps.contain(canvas,(1200,3500)).save(OUT/'figure_all_cases_web.jpg',quality=94)
    summary=dict(cases=display,methods=LABELS,registered_cameras_only=True,training_steps=0,ply_writes=0,
        input_difference=INPUT_NOTE,source_files_unchanged=True,case_results={})
    for case in display:
        audit=r.read(OUT/case/'camera_matching.json');instance=r.read(OUT/case/'instance_matching.json')
        assert audit['source_files_unchanged']
        for m in audit['methods'].values():assert m['localization_camera']==m['final_camera']==m['before_camera']==m['after_camera']
        summary['case_results'][case]=dict(reference=audit['reference_camera']['name'],instance_relations=instance['against_ours'],
            methods={m:dict(camera=x['camera']['name'],same_world_camera=x['same_world_camera_as_ours'],
            relative_view_difference=x['relative_view_difference'],visibility=x['metrics']['new_visibility'],
            old_visibility=x['metrics']['old_visibility'],angle_constraints_relaxed=x['angle_constraints_relaxed']) for m,x in audit['methods'].items()})
    r.save(OUT/'summary.json',summary)
    caption='Localization and final 3D editing in an industrial scene. All panels are Gaussian rasterizations of existing scenes. Within each column, localization and final editing share identical registered COLMAP camera intrinsics/extrinsics and resolution. Shared source-Gaussian identity and 3D regions determine same-instance relations; a common world camera is preferred when readable. Different instances are viewed from comparable relative directions where practical, with visibility and source-scene quality taking priority when matching is infeasible. Localization overlays use 25% red weighted by actual selected-Gaussian contribution, with contours; all saved selection indices are retained. Baselines use their native localization pipelines. Ours additionally receives a reference photograph to specify the target instance, so localization inputs are not identical. GSLE uses its original native DI-SDS final result. Unsuccessful edits and erroneous instance selections are preserved. Before/after crops are identical within a method and cover the primary old region and generated asset; full-frame localization differences remain in the main figure. No Gaussian PLY, target location, or model was changed.'
    (OUT/'figure_caption.txt').write_text(caption)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SCI 定性对比：COLMAP 视角与实例匹配</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px}img{max-width:100%}section{margin:30px 0}</style><h1>SCI 定性对比：真实相机、实例匹配、上下同视角</h1><p>全部展示相机来自 350 个真实 COLMAP 注册相机；Ours 参考视角也重新选择。先判断实际三维目标实例，能共享世界相机时优先共享；不同实例尽量匹配观察方向，无法兼顾时保留清楚的有效观察视角。红色定位为 25% 覆盖加轮廓线。</p><p><strong>输入差异：</strong>基线使用原生定位，Ours 额外提供参考照片指定目标实例。不是完全相同输入信息的定位比较。</p><p><a href="figure_all_cases.png">高清总览</a> | <a href="summary.json">汇总记录</a> | <a href="figure_caption.txt">英文图注</a></p>'''
    for case in display:
        page+=f'<section><h2><a href="{case}/index.html">{case}：主图、Before/After 与候选相机</a></h2><a href="{case}/figure_localization_final.png"><img loading="lazy" src="{case}/figure_localization_final.png"></a></section>'
    (OUT/'index.html').write_text(page+'</html>',encoding='utf-8')
    with zipfile.ZipFile(OUT/'viewpoint_aligned_v2_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.png','.jpg','.json','.html','.txt'):z.write(p,p.relative_to(OUT))
    print('PACKAGED',display,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--package',nargs='*');a=p.parse_args()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45);OUT.mkdir(parents=True,exist_ok=True)
    if a.case:run(a.case)
    if a.package is not None:package(a.package or lr.CASES)
