"""Uniform scale + ground-preserving rigid placement, fitted to anchor silhouette."""
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .scene import read_vertices, save_vertices
from .context import write_json


def xyz_of(vertices):
    return np.column_stack([vertices[k] for k in ('x','y','z')])


def transform_vertices(vertices, scale, rotation, translation):
    assert scale > 0 and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
    output = vertices.copy()
    xyz = xyz_of(vertices) @ rotation.T * scale + translation
    normal = np.column_stack([vertices[k] for k in ('nx','ny','nz')]) @ rotation.T
    quat = np.column_stack([vertices[f'rot_{i}'] for i in range(4)])[:,[1,2,3,0]]
    quat = (Rotation.from_matrix(rotation)*Rotation.from_quat(quat)).as_quat()[:,[3,0,1,2]]
    for i,k in enumerate(('x','y','z')): output[k]=xyz[:,i]
    for i,k in enumerate(('nx','ny','nz')): output[k]=normal[:,i]
    for i in range(3): output[f'scale_{i}']+=np.log(scale)
    for i in range(4): output[f'rot_{i}']=quat[:,i]
    return output


def fit(vertices, frame, camera, mask_path, config, destination):
    points = xyz_of(vertices)
    # Deterministic surface subset is sufficient for ordinary pose fitting.
    probe = points[::max(1,len(points)//40000)]
    axes = np.asarray(frame['axes']); center=np.asarray(frame['center'])
    low, high = np.asarray(frame['local_bbox_low']), np.asarray(frame['local_bbox_high'])
    extent = high-low
    with Image.open(mask_path) as im:
        truth = np.asarray(im.convert('L').resize((192,192),Image.Resampling.NEAREST))>127
    if not truth.any(): raise ValueError('Empty reference mask')
    R=np.asarray(camera['R']); t=np.asarray(camera['t'])
    fx,fy=camera['fx']*192/camera['width'],camera['fy']*192/camera['height']
    cx,cy=camera['cx']*192/camera['width'],camera['cy']*192/camera['height']
    def placement(params, cloud):
        yaw, fraction, dx, dz=params
        local_rot=Rotation.from_euler('y',yaw,degrees=True).as_matrix()
        rotated=points@local_rot.T
        lo,hi=rotated.min(0),rotated.max(0)
        scale=float(np.min(extent/np.maximum(hi-lo,1e-8))*fraction)
        offset=(low+high)/2-scale*(lo+hi)/2
        offset[1]=low[1]-scale*lo[1]
        offset[[0,2]]+=np.array([dx,dz])*extent[[0,2]]
        rot=axes@local_rot
        shift=center+axes@offset
        return cloud@rot.T*scale+shift,scale,rot,shift
    def silhouette(params):
        world,scale,rot,shift=placement(params,probe)
        cam=world@R.T+t
        valid=cam[:,2]>1e-6
        xy=cam[valid,:2]/cam[valid,2:]*[fx,fy]+[cx,cy]
        xy=np.rint(xy).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,0]<192)&(xy[:,1]>=0)&(xy[:,1]<192)
        mask=np.zeros((192,192),np.uint8)
        mask[xy[valid,1],xy[valid,0]]=1
        return cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(bool)
    def loss(params):
        mask=silhouette(params)
        iou=np.count_nonzero(mask&truth)/max(1,np.count_nonzero(mask|truth))
        world,_,_,_=placement(params,probe)
        local=(world-center)@axes
        outside=np.maximum(low-local,0)+np.maximum(local-high,0)
        return 1-iou+5*np.mean((outside/extent)**2)
    candidates=[]
    for yaw in np.arange(-180,180,15):
        for fraction in (.65,.8,config['fit_fraction']):
            p=[float(yaw),fraction,0,0]
            candidates.append((loss(p),p))
    candidates.sort(key=lambda x:x[0])
    best=candidates[0]
    for _,initial in candidates[:3]:
        result=minimize(loss,initial,method='Powell',bounds=[(-180,180),(.25,1),(-.35,.35),(-.35,.35)],
                        options={'maxiter':60,'xtol':.002,'ftol':.0001})
        if result.fun<best[0]: best=(float(result.fun),result.x.tolist())
    _,scale,rotation,translation=placement(best[1],points)
    world=transform_vertices(vertices,scale,rotation,translation)
    save_vertices(destination/'object_world.ply',world)
    matrix=np.eye(4); matrix[:3,:3]=scale*rotation; matrix[:3,3]=translation
    write_json(destination/'transform.json',{'scale':scale,'rotation':rotation.tolist(),
        'translation':translation.tolist(),'object_to_world':matrix.tolist(),
        'world_to_object':np.linalg.inv(matrix).tolist(), 'fit_parameters':best[1],
        'silhouette_objective':best[0], 'floor_local_y':float(low[1]),
        'method':'uniform-scale Y-up yaw and floor-anchored translation, automatic anchor silhouette fit',
        'reference_mask':str(mask_path),'camera':camera})
    preview=np.zeros((192,192,3),np.uint8)
    preview[truth,1]=255; preview[silhouette(best[1]),0]=255
    Image.fromarray(preview).save(destination/'alignment_overlay.png')
    return world
