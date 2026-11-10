"""Display-only GE case04 camera search: native localization, Ours-like direction."""
import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
import render as r


def score_record(item):
    loc=item['localization'];gen=item['generated_object'];pixels=item['full_selection_visible_pixels']
    score=30*max(0,.97-loc['capture'])+10*max(0,.65-loc['visibility'])
    score+=2.2*abs(math.log(max(loc['size'],.005)/.48))
    score+=4*max(0,.96-gen['capture'])+2*max(0,.65-gen['visibility'])
    score+=1.2*max(0,.18-gen['size'])+2*max(0,1-pixels/250)
    score+=1.5*max(0,.65-loc['scene_coverage'])+.065*abs(item['yaw_offset'])+.035*abs(item['pitch_offset'])
    # Require room at the image edge for both localization and generated target.
    # Guard-canvas alpha mass alone can underestimate perceptual edge clipping.
    for stats in [loc,gen]:
        if stats['bbox']:
            x0,y0,x1,y1=stats['bbox'];margin=min(x0,y0,224-x1,224-y1)
            score+=2*max(0,1-margin/8)
    return score


@torch.no_grad()
def choose(original, final, ours, ours_center, up, out):
    folder=r.BASE/'gaussianeditor_native/case04'
    ids=np.load(folder/'localization/selected_indices.npy')
    points=original.xyz[torch.as_tensor(ids,device='cuda')].cpu().numpy()
    # Identify the native-primary-visible component strictly for camera framing.
    # Every saved index remains in the eventual red localization render.
    all_xyz=original.xyz.cpu().numpy()
    radius=float(np.linalg.norm(np.diff(np.quantile(all_xyz,[.005,.995],axis=0),axis=0))*.003)
    dist,nn=cKDTree(points).query(points,k=min(32,len(points)),distance_upper_bound=radius)
    rr=np.broadcast_to(np.arange(len(points))[:,None],nn.shape);valid=nn<len(points)
    graph=coo_matrix((np.ones(valid.sum()),(rr[valid],nn[valid])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False);counts=np.bincount(labels)
    native=r.cpu_camera(r.read(folder/'anchor_camera.json')).resize(256,256)
    component_scores=[]
    top=np.argsort(counts)[-12:][::-1]
    for start in range(0,len(top),3):
        packed=torch.zeros_like(original.xyz)
        for j,k in enumerate(top[start:start+3]):packed[torch.as_tensor(ids[labels==k],device='cuda'),j]=1
        scores=r.c.raster(native,original,packed).sum((1,2)).cpu().numpy()
        for j,k in enumerate(top[start:start+3]):
            component_scores.append(dict(label=int(k),count=int(counts[k]),native_visible_mass=float(scores[j])))
    component=max(component_scores,key=lambda x:x['native_visible_mass'])
    focus_ids=ids[labels==component['label']]
    np.save(out/'ge_localization_framing_indices.npy',focus_ids)
    focus=original.subset(torch.as_tensor(focus_ids,device='cuda'))
    loc_center=np.median(points[labels==component['label']],axis=0)
    obj=r.c.GaussianData.load(folder/'add/object_world.ply')
    obj_center=np.median(obj.xyz.cpu().numpy(),axis=0)
    previous=r.read(r.BASE/'matched_views/case04/camera_params.json')['methods']['gaussianeditor']['target']
    verified,vis_scene,final_flags=r.flag_data(dict(previous),final,original)
    del verified
    source_flags=torch.zeros_like(original.xyz);source_flags[:,1]=1
    source_flags[torch.as_tensor(focus_ids,device='cuda'),0]=1
    full_flags=torch.zeros_like(original.xyz);full_flags[torch.as_tensor(ids,device='cuda')]=1
    direction=r.c.normalized(ours.center-ours_center)
    local_xyz=focus.xyz.cpu().numpy();object_xyz=obj.xyz.cpu().numpy()
    support=np.concatenate([np.quantile(local_xyz,[.005,.995],axis=0),np.quantile(object_xyz,[.005,.995],axis=0)])
    bounds=r.c.bbox_corners(support.min(0),support.max(0))
    def camera(center,distance,yaw,pitch):
        q=Rotation.from_rotvec(up*np.radians(yaw)).as_matrix()
        right=r.c.normalized(np.cross(up,q@direction))
        q=Rotation.from_rotvec(right*np.radians(-pitch)).as_matrix()@q
        position=center+distance*(q@direction)
        cam=r.c.look_at(position,center,up,resolution=1024,name='case04_ge_localization_visible')
        return r.c.Camera(cam.R,cam.t,1024,1024,ours.fx,ours.fy,512,512,cam.name)
    trials=[]
    def evaluate(cam,center,yaw,pitch,kind):
        low=cam.resize(224,224)
        loc=r.candidate_metrics(cam,focus,original,source_flags,res=224)
        gen=r.candidate_metrics(cam,obj,final,final_flags,res=224)
        all_loc=r.c.raster(low,original,full_flags)[0]
        pixels=int((all_loc>.02).sum())
        # Old-device localization is primary. Generated-object visibility remains
        # a secondary criterion; physical placement errors must stay unchanged.
        score=30*max(0,.97-loc['capture'])+10*max(0,.65-loc['visibility'])
        score+=2.2*abs(math.log(max(loc['size'],.005)/.48))
        score+=4*max(0,.96-gen['capture'])+2*max(0,.65-gen['visibility'])
        score+=1.2*max(0,.18-gen['size'])+2*max(0,1-pixels/250)
        score+=1.5*max(0,.65-loc['scene_coverage'])+.025*abs(yaw)+.035*abs(pitch)
        record=dict(camera=cam.record(),center=center.tolist(),kind=kind,score=score,
            yaw_offset=yaw,pitch_offset=pitch,localization=loc,generated_object=gen,
            full_selection_visible_pixels=pixels)
        record['score']=score_record(record)
        trials.append(record)
        return record
    # Both center choices remain derived exclusively from GE's own artifacts.
    for fraction in [0.,.5]:
        center=(1-fraction)*loc_center+fraction*obj_center
        lo,hi=.05,30.
        for _ in range(25):
            mid=(lo+hi)/2;cam=camera(center,mid,0,0);xy,z=cam.project(bounds)
            size=np.ptp(xy,axis=0).max()/1024 if (z>0).all() else 1e6
            if size>.72:lo=mid
            else:hi=mid
        initial=(lo+hi)/2
        for factor in [.75,1.,1.35]:
            for yaw,pitch in [(0,0),(-12,0),(12,0),(-24,0),(24,0),(0,8),(0,-8)]:
                evaluate(camera(center,initial*factor,yaw,pitch),center,yaw,pitch,
                         'localization_center' if fraction==0 else 'localization_generation_midpoint')
    best=min(trials,key=lambda x:x['score'])
    ranked=sorted(trials,key=lambda x:x['score'])[:6]
    for i,item in enumerate(ranked):
        cam=r.c.Camera.from_record(item['camera']);colors=original.colors(cam.center)
        idx=torch.as_tensor(ids,device='cuda');colors[idx]=.4*colors[idx]+.6*colors.new_tensor([1.,0.,0.])
        r.c.save_image(out/f'candidate_{i}_localization.png',r.c.raster(cam,original,colors))
        r.c.save_image(out/f'candidate_{i}_final.png',r.c.render(cam,final,aux=False)['rgb'])
    cam=r.c.Camera.from_record(best['camera'])
    delta=r.orientation(-cam.R[2],up);reference=r.orientation(-ours.R[2],up)
    report=dict(selected=best,candidates=trials,native_reference=native.record(),
        native_components=component_scores,framing_component=component,
        localization_center=loc_center.tolist(),generated_center=obj_center.tolist(),
        full_localization_count=len(ids),framing_count=len(focus_ids),
        camera_optical_yaw_difference=r.delta_angle(delta['yaw'],reference['yaw']),
        camera_optical_pitch_difference=delta['pitch']-reference['pitch'],
        policy='Camera framing only: native-visible localization component; all original selected indices remain rendered. Favor Ours-like direction subject to real localization visibility. No Gaussian changes.')
    r.save(out/'ge_camera_search.json',report)
    return cam,report
