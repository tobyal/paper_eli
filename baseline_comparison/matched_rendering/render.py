"""Read-only Viewpoint-Matched Rendering of independently localized final PLYs.

Baseline target centers are obtained exclusively from their own artifacts. Ours
supplies viewing direction, intrinsics, framing and approximate projected size.
No scene is transformed, filtered, optimized, or saved back to PLY.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(os.environ.get('ROOT','/mnt/sda/ubuntu/lkx/papers/paper_eli'))
BASE=ROOT/'baseline_comparison'
sys.path.insert(0,str(BASE/'vendor/gsle_native'))
import common as c
METHODS=['original','gaussianeditor','dge','gsle','ours']
LABELS=['Original','GaussianEditor','DGE','GSLE','Ours']
FIELDS=['x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    payload=json.dumps(value,indent=2,ensure_ascii=False,default=encode)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(payload,encoding='utf-8');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def xyz(v):return np.stack([v[k] for k in ('x','y','z')],1)
def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-60,60)))
def geometry(v):
    p=xyz(v);a=alpha(v);valid=np.isfinite(p).all(1)&(a>.01)
    assert valid.sum()>3
    p=p[valid];s=np.exp(np.stack([v[k][valid] for k in ('scale_0','scale_1','scale_2')],1)).max(1)
    lo=np.quantile(p-2*s[:,None],.005,axis=0);hi=np.quantile(p+2*s[:,None],.995,axis=0)
    return dict(median=np.median(p,axis=0).tolist(),bounds=[lo.tolist(),hi.tolist()],
        valid_opacity_points=int(valid.sum()),count=len(v),radius=float(np.linalg.norm(hi-lo)/2))
def corners(meta):return c.bbox_corners(*np.asarray(meta['bounds']))
def cpu_camera(r):
    if 't' in r:return c.Camera.from_record(r)
    return c.Camera(np.array(r['R']).T,np.array(r['T']),r['width'],r['height'],
        r['width']/(2*np.tan(r['FoVx']/2)),r['height']/(2*np.tan(r['FoVy']/2)),name=r['name'])
def orientation(direction,up):
    direction=c.normalized(direction);north=c.normalized(np.array([0.,0.,1.])-up*up[2])
    east=c.normalized(np.cross(up,north))
    return dict(yaw=float(np.degrees(np.arctan2(direction@east,direction@north))),
        pitch=float(np.degrees(np.arcsin(np.clip(direction@up,-1,1)))))
def delta_angle(a,b):return float((a-b+180)%360-180)

def camera_at(ref,ref_center,center,distance,up,yaw=0.,pitch=0.,name='matched'):
    """Rigidly transport the camera rig; retain optical offset, roll and K.

    For off-center intrinsics, the target ray matches Ours' normalized image
    location rather than incorrectly forcing the optical axis to image center.
    """
    direction=c.normalized(ref.center-ref_center)
    yaw_rot=Rotation.from_rotvec(up*np.radians(yaw)).as_matrix()
    right=c.normalized(np.cross(up,yaw_rot@direction))
    pitch_rot=Rotation.from_rotvec(right*np.radians(-pitch)).as_matrix()
    Q=pitch_rot@yaw_rot
    pos=np.asarray(center)+distance*(Q@direction)
    R=ref.R@Q.T
    return c.Camera(R,-R@pos,ref.width,ref.height,ref.fx,ref.fy,ref.cx,ref.cy,name)

def alpha_stats(a):
    mask=a>.02;yy,xx=np.where(mask)
    if not len(xx):return dict(mass=float(a.sum()),size=0.,bbox=None,pixels=0)
    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    return dict(mass=float(a.sum()),size=max(box[2]-box[0],box[3]-box[1])/max(a.shape),
        bbox=box,pixels=len(xx),centroid=[float((xx+.5).mean()/a.shape[1]),float((yy+.5).mean()/a.shape[0])])

@torch.no_grad()
def object_alpha(cam,obj):
    return c.raster(cam,obj,torch.ones_like(obj.xyz))[0].cpu().numpy()

def dge_focus(ids,source_v,source_data,folder,out):
    """Choose a framing component by native primary-view contribution only.

    The full native selection is retained and reported; no segmentation or
    final scene changes. This avoids aiming at empty space between unrelated
    selected devices. Native primary view, never Ours, determines the focus.
    """
    pts=xyz(source_v[ids]);scene_bounds=np.quantile(xyz(source_v),[.005,.995],axis=0)
    radius=float(np.linalg.norm(scene_bounds[1]-scene_bounds[0])*.003)
    dist,nn=cKDTree(pts).query(pts,k=min(32,len(pts)),distance_upper_bound=radius)
    rr=np.broadcast_to(np.arange(len(pts))[:,None],nn.shape);valid=nn<len(pts)
    graph=coo_matrix((np.ones(valid.sum()),(rr[valid],nn[valid])),shape=(len(pts),len(pts)))
    count,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels);top=np.argsort(counts)[-6:][::-1]
    native=cpu_camera(read(folder/'records/localization.json')['review_cameras'][0])
    components=[]
    for start in range(0,len(top),3):
        batch=top[start:start+3];colors=torch.zeros_like(source_data.xyz)
        for j,k in enumerate(batch):colors[torch.as_tensor(ids[labels==k],device='cuda'),j]=1
        with torch.no_grad():scores=c.raster(native,source_data,colors).sum((1,2)).cpu().numpy()
        for j,k in enumerate(batch):
            components.append(dict(label=int(k),count=int(counts[k]),native_visible_contribution=float(scores[j]),
                center=np.median(pts[labels==k],axis=0).tolist()))
    selected=max(components,key=lambda x:x['native_visible_contribution'])
    focus=ids[labels==selected['label']]
    np.save(out/'dge_display_focus_indices.npy',focus)
    record=dict(policy='Camera framing only: native-primary-visible dominant connected component; full final scene and all native selection remain unchanged.',
        native_primary_camera=native.record(),native_selected_count=len(ids),component_count=count,
        radius=radius,radius_rule='0.003 * original-scene 0.5%-99.5% bounding-box diagonal',
        components_ranked=components,selected=selected,focus_fraction=len(focus)/len(ids),
        all_selected_geometry=geometry(source_v[ids]),indices_source=str(folder/'target_indices.npy'))
    return focus,record

def target_info(method,row,source_v,source_data,out):
    if method=='ours':
        p=Path(row['ours_scene']);obj=p.parent/'object_world.ply';target=read(row['target_record'])
        v=PlyData.read(str(obj))['vertex'].data
        return dict(center=target['center'],geometry=geometry(v),object_path=str(obj),scene_path=str(p),
            center_source=row['target_record'],center_rule='Unchanged Ours M1 target center',
            source_records=[row['target_record'],row['ours_anchor']],ours_guard_passed=row.get('ours_guard_passed'))
    folder=BASE/(method+'_native')/row['case'];p=folder/'final_scene.ply'
    if method=='dge':
        ids=np.load(folder/'target_indices.npy')
        focus,notes=dge_focus(ids,source_v,source_data,folder,out)
        g=geometry(source_v[focus])
        return dict(center=g['median'],geometry=g,object_path=None,indices=focus.tolist(),scene_path=str(p),
            center_source=str(folder/'target_indices.npy'),center_rule='Median of native-primary-visible tracing component; no Ours coordinates',
            component_framing=notes,visibility_proxy='Original traced component in original scene; final DGE Gaussian index identity is not assumed.',
            source_records=[str(folder/'target_indices.npy'),str(folder/'records/localization.json')])
    if method=='gsle':
        target=read(folder/'target_bbox.json');obj=folder/'refine/final_object_world.ply'
        g=geometry(PlyData.read(str(obj))['vertex'].data)
        return dict(center=target['center'],geometry=g,object_path=str(obj),scene_path=str(p),
            center_source=str(folder/'target_bbox.json'),center_rule='Frozen native LangSplat target center; final-object expansion affects framing distance only',
            source_records=[str(folder/'target_bbox.json')],native_target_bbox=target['bbox_corners'])
    obj=folder/'add/object_world.ply'
    if obj.exists():
        g=geometry(PlyData.read(str(obj))['vertex'].data)
        return dict(center=g['median'],geometry=g,object_path=str(obj),scene_path=str(p),
            center_source=str(obj),center_rule='Median of native Add world-space object, cross-referenced with native placement; no correction to placement',
            placement=read(folder/'add/placement.json'),source_records=[str(folder/'add/placement.json'),str(folder/'records/add_region.json')])
    obj=folder/'selected_target.ply';g=geometry(PlyData.read(str(obj))['vertex'].data)
    return dict(center=g['median'],geometry=g,object_path=str(obj),scene_path=str(p),fallback=True,
        center_source=str(obj),center_rule='Fallback: own Semantic Tracing target, Add object unavailable',
        source_records=[str(folder/'records/add_region.json')])

def flag_data(info,scene,source_data):
    if info.get('indices') is not None:
        idx=torch.as_tensor(info['indices'],device='cuda');obj=source_data.subset(idx)
        colors=torch.zeros_like(source_data.xyz);colors[:,1]=1;colors[idx,0]=1
        return obj,source_data,colors
    obj=c.GaussianData.load(info['object_path'])
    # Exact final-tail identity supports contribution measurement in the real scene.
    ov=PlyData.read(info['object_path'])['vertex'].data
    sv=PlyData.read(info['scene_path'])['vertex'].data
    equal=all(np.array_equal(ov[k],sv[-len(ov):][k]) for k in FIELDS)
    assert equal,'Cannot assume target identity in final PLY'
    info['final_object_tail_verified']=True
    colors=torch.zeros_like(scene.xyz);colors[:,1]=1;colors[-len(obj.xyz):,0]=1
    return obj,scene,colors

@torch.no_grad()
def candidate_metrics(cam,obj,vis_scene,colors,res=256):
    low=cam.resize(res,res)
    a=object_alpha(low,obj)
    visible=c.raster(low,vis_scene,colors).cpu().numpy()
    guard=c.Camera(low.R,low.t,res*2,res*2,low.fx,low.fy,low.cx+res/2,low.cy+res/2,'framing_guard')
    ga=object_alpha(guard,obj)
    stats=alpha_stats(a)
    stats.update(visibility=float(visible[0].sum()/max(a.sum(),1e-8)),
        capture=float(a.sum()/max(ga.sum(),1e-8)),scene_coverage=float((visible[1]>.1).mean()))
    return stats

def match(ref,ref_center,info,up,desired,obj,vis_scene,colors,label):
    center=np.asarray(info['center']);distance=np.linalg.norm(ref.center-ref_center)
    box=corners(info['geometry']);lo,hi=distance*.05,distance*20
    # Geometric distance initialization only; selection uses real Gaussian alpha.
    for _ in range(30):
        mid=(lo+hi)/2;cam=camera_at(ref,ref_center,center,mid,up)
        uv,z=cam.project(box)
        size=np.ptp(uv,axis=0).max()/ref.width if (z>0).all() else 1e6
        if size>desired:lo=mid
        else:hi=mid
    initial=(lo+hi)/2
    proposals=[]
    same=np.linalg.norm(center-ref_center)<.08*max(info['geometry']['radius'],1e-6)
    if same:proposals.append((ref,0.,0.,True))
    for scale in [.7,1.,1.4,2.]:proposals.append((camera_at(ref,ref_center,center,initial*scale,up,name=label),0.,0.,False))
    trials=[]
    def evaluate(proposal):
        cam,yaw,pitch,exact=proposal;m=candidate_metrics(cam,obj,vis_scene,colors)
        score=50*max(0,.985-m['capture'])+8*max(0,.80-m['visibility'])+2*abs(math.log(max(m['size'],.001)/max(desired,.001)))
        score+=3*max(0,.7-m['scene_coverage'])+.09*(abs(yaw)/8+abs(pitch)/5)
        if exact:score-=.12
        rec=dict(camera=cam.record(),yaw_offset=yaw,pitch_offset=pitch,exact_reference=exact,score=score,**m)
        trials.append(rec);return rec
    for proposal in proposals:evaluate(proposal)
    best=min(trials,key=lambda x:x['score'])
    if best['visibility']<.85 or best['capture']<.98:
        for yaw,pitch in [(-8,0),(8,0),(-15,0),(15,0),(0,6),(0,-6),(-8,6),(8,6)]:
            evaluate((camera_at(ref,ref_center,center,initial,up,yaw,pitch,label),yaw,pitch,False))
    best=min(trials,key=lambda x:x['score'])
    if best['capture']<.97:
        cam=c.Camera.from_record(best['camera']);rad=np.linalg.norm(cam.center-center)
        for factor in [1.5,2.]:
            evaluate((camera_at(ref,ref_center,center,rad*factor,up,best['yaw_offset'],best['pitch_offset'],label),best['yaw_offset'],best['pitch_offset'],False))
        best=min(trials,key=lambda x:x['score'])
    return c.Camera.from_record(best['camera']),dict(selected=best,candidates=trials,
        desired_object_size=desired,initial_distance=initial,same_location_preference=same)

def square_crop(box,size,padding=1.22):
    if box is None:return [0,0,size,size]
    x0,y0,x1,y1=box;side=min(size,max(x1-x0,y1-y0)*padding)
    x=max(0,min(size-side,(x0+x1-side)/2));y=max(0,min(size-side,(y0+y1-side)/2))
    return [int(x),int(y),int(math.ceil(x+side)),int(math.ceil(y+side))]

def board(path,rows,title,cell=500):
    cols=len(rows[0]);header=80;rh=cell+40
    canvas=Image.new('RGB',(cols*cell,header+len(rows)*rh),'white');d=ImageDraw.Draw(canvas)
    titlefont=ImageFont.truetype('DejaVuSans.ttf',27);font=ImageFont.truetype('DejaVuSans.ttf',24)
    d.text((16,12),title,font=titlefont,fill='black')
    caption='Viewpoint-Matched Rendering | Independent target locations; camera positions may differ'
    size=19
    while ImageFont.truetype('DejaVuSans.ttf',size).getlength(caption)>cols*cell-32:size-=1
    d.text((16,47),caption,font=ImageFont.truetype('DejaVuSans.ttf',size),fill='#444444')
    for i,row in enumerate(rows):
        for j,(label,p,box) in enumerate(row):
            x,y=j*cell,header+i*rh;d.text((x+10,y+7),label,font=font,fill='black')
            im=Image.open(p).convert('RGB')
            if box:im=im.crop(box)
            im=ImageOps.contain(im,(cell,cell),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+(cell-im.width)//2,y+40+(cell-im.height)//2))
    canvas.save(path,dpi=(300,300))

def make_figures(out,case,report):
    board(out/'comparison_main.png',[[(l,out/(m+'.png'),None) for m,l in zip(METHODS,LABELS)]],case+' | Main view')
    board(out/'comparison_crops.png',[[(l,out/(m+'.png'),report['crops'][m]) for m,l in zip(METHODS,LABELS)]],case+' | Target-centered crops')
    board(out/'comparison_multiview.png',[[(l+f' / aux {i}',out/'views'/f'aux{i}'/(m+'.png'),None) for m,l in zip(METHODS,LABELS)] for i in (1,2)],case+' | Auxiliary views')
    board(out/'baseline_before_after.png',[[(m+' / Original',out/('original_'+m+'.png'),None),(m+' / Final',out/(m+'.png'),None)] for m in ('gaussianeditor','dge','gsle')],case+' | Each baseline: same-camera before / after')
    page='<!doctype html><html><meta charset="utf-8"><title>'+case+' Matched views</title><style>body{font:17px/1.6 system-ui;max-width:1700px;margin:auto;padding:20px;background:#f5f6f8}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{background:white;padding:16px;margin:20px 0}</style><h1>'+case+' | Viewpoint-Matched Rendering</h1><p>Different methods may use different camera positions around independently selected targets. Original and Ours use the saved Ours anchor. Every panel is rendered from an unchanged final Gaussian PLY. GSLE uses the original native DI-SDS final result, not Coarse or the color-only experiment.</p><p><a href="camera_params.json">Camera parameters</a> | <a href="matching_report.json">Matching report and source hashes</a></p>'
    for file,label in [('comparison_main.png','Main view'),('comparison_crops.png','Local crops'),('comparison_multiview.png','Two auxiliary views'),('baseline_before_after.png','Baseline-specific original scene controls')]:
        page+='<section><h2>'+label+'</h2><a href="'+file+'"><img src="'+file+'" loading="lazy"></a></section>'
    page+='<h2>Method-specific observations</h2><pre>'+json.dumps(report['notes'],ensure_ascii=False,indent=2)+'</pre></html>'
    (out/'index.html').write_text(page,encoding='utf-8')

@torch.no_grad()
def run(case,res):
    start=time.time();spec=read(BASE/'prompts.json');row=next(x for x in spec['cases'] if x['case']==case)
    out=BASE/'matched_views'/case;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'matching_report.json').exists(),'Completed rendering exists; preserve it before rerunning'
    source_v=PlyData.read(spec['source_scene'])['vertex'].data
    original=c.GaussianData.load(spec['source_scene'])
    target=read(row['target_record']);center=np.asarray(target['center']);up=c.normalized(np.array(target['up']))
    saved_ref=c.Camera.from_record(read(row['ours_anchor']));ref=saved_ref.resize(res,res)
    refs=[ref]+[camera_at(ref,center,center,np.linalg.norm(ref.center-center),up,angle,0,f'ours_aux{n}') for n,angle in [(1,-20),(2,20)]]
    report=dict(case=case,label='Viewpoint-Matched Rendering',resolution=[res,res],
        methodology='Independent target centers; exact Ours anchor and two +/-20 degree yaw auxiliary rigs. Camera-only distance and bounded yaw/pitch matching. No Gaussian edits.',
        renderer='Pinned gsle_native.common: diff_gaussian_rasterization, full stored SH, normalized quaternion, sigmoid opacity, exp scale, black background; identical settings for all methods.',
        source_manifest=str(BASE/'prompts.json'),gsle_variant='gsle_native original DI-SDS final',notes={},sources={},crops={})
    cameras=dict(reference_camera_original=saved_ref.record(),reference_camera_resized=ref.record(),
        ours_target_center=center.tolist(),world_up=up.tolist(),target_pixel_policy='Preserve Ours target ray, optical-axis offset, roll and off-center intrinsics. A centered principal point is never substituted.',methods={})
    input_paths={Path(spec['source_scene']),BASE/'prompts.json',Path(row['ours_anchor']),Path(row['target_record'])}
    input_before={str(p):sha(p) for p in input_paths}
    ours=target_info('ours',row,source_v,original,out)
    obj=c.GaussianData.load(ours['object_path'])
    desired=[]
    for refcam in refs:
        stats=alpha_stats(object_alpha(refcam.resize(256,256),obj));desired.append(float(np.clip(stats['size'],.2,.7)))
    del obj
    for method in ['ours','gaussianeditor','dge','gsle']:
        info=ours if method=='ours' else target_info(method,row,source_v,original,out)
        input_paths.add(Path(info['scene_path']))
        if info.get('object_path'):input_paths.add(Path(info['object_path']))
        input_paths.update(Path(x) for x in info['source_records'])
        for p in input_paths:
            if str(p) not in input_before:input_before[str(p)]=sha(p)
        info_public={k:v for k,v in info.items() if k!='indices'}
        scene=c.GaussianData.load(info['scene_path'])
        obj,vis_scene,colors=flag_data(info,scene,original)
        info_public['final_object_tail_verified']=info.get('final_object_tail_verified',False)
        method_cameras=[];matches=[]
        quality_path=BASE/(method+'_native')/case/'records/visual_review.json'
        report['notes'][method]=read(quality_path) if quality_path.exists() else dict(ours_source_guard_passed=row.get('ours_guard_passed'),source_selection='Exact ours_scene from prompts.json, including user-approved yaw correction where present; no substitution with guarded fallback.')
        for i,refcam in enumerate(refs):
            label='main' if i==0 else f'aux{i}';folder=out/'views'/label;folder.mkdir(parents=True,exist_ok=True)
            if method=='ours':cam=refcam;selection=dict(policy='Unmodified saved Ours anchor / transported auxiliary rig')
            else:cam,selection=match(refcam,center,info,up,desired[i],obj,vis_scene,colors,method+'_'+label)
            print(case,method,label,'camera ready',flush=True)
            c.save_image(folder/(method+'.png'),c.render(cam,scene,aux=False)['rgb'])
            c.save_image(folder/('original_'+method+'.png'),c.render(cam,original,aux=False)['rgb'])
            a=object_alpha(cam,obj);metrics=alpha_stats(a);c.save_image(folder/(method+'_target_alpha.png'),torch.from_numpy(a)[None])
            if method=='ours':
                c.save_image(folder/'original.png',c.render(cam,original,aux=False)['rgb'])
            oc=np.asarray(info['center']);relative=orientation(cam.center-oc,up);template=orientation(refcam.center-center,up)
            uv,z=cam.project(oc[None]);rr=dict(view=label,target_center=info['center'],reference_camera=refcam.record(),
                matched_camera=cam.record(),camera_position=cam.center.tolist(),distance=float(np.linalg.norm(cam.center-oc)),
                yaw=relative['yaw'],pitch=relative['pitch'],yaw_difference=delta_angle(relative['yaw'],template['yaw']),
                pitch_difference=relative['pitch']-template['pitch'],target_pixel_normalized=(uv[0]/res).tolist(),
                nominal_fov_degrees=[float(np.degrees(2*np.arctan(res/(2*f)))) for f in [cam.fx,cam.fy]],
                final_object_projection=metrics,selection=selection)
            method_cameras.append(rr)
            if i==0:
                for name in [method, 'original_'+method]:
                    import shutil;shutil.copy2(folder/(name+'.png'),out/(name+'.png'))
                report['crops'][method]=square_crop(metrics['bbox'],res)
                if method=='ours':
                    shutil.copy2(folder/'original.png',out/'original.png');report['crops']['original']=report['crops']['ours']
            matches.append(selection)
        cameras['methods'][method]=dict(target=info_public,views=method_cameras)
        report['sources'][method]=dict(scene=info['scene_path'],target=info_public)
        if method=='dge':report['notes'][method]['framing_disclosure']=info['component_framing']
        del obj,vis_scene,colors,scene;torch.cuda.empty_cache()
    del original;torch.cuda.empty_cache()
    # All input paths are read-only. Exact content hashes and file stats are saved.
    report['input_hashes']={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(input_paths)}
    report['input_files_unchanged']=all(r['sha256']==input_before[p] for p,r in report['input_hashes'].items())
    assert report['input_files_unchanged']
    report['no_scene_writes']=True;report['training_steps']=0;report['seconds']=time.time()-start
    save(out/'camera_params.json',cameras);save(out/'matching_report.json',report)
    make_figures(out,case,report)
    print('COMPLETE',case,report['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--resolution',type=int,default=1024);a=p.parse_args()
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    run(a.case,a.resolution)
