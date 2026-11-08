"""Independent triangle ray intersections verify Blender/CV frame and EXR units."""
import argparse
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image
from industrial_swap_v2.core.context import read_json,write_json
from industrial_swap_v2.core.pbr_math import read_depth_exr


def validate(folder):
    record=read_json(folder/'cameras.json')
    loaded=trimesh.load(record['glb'],force='scene',process=False)
    meshes=[]
    for node in loaded.graph.nodes_geometry:
        matrix,key=loaded.graph[node];mesh=loaded.geometry[key].copy();mesh.apply_transform(matrix);meshes.append(mesh)
    mesh=trimesh.util.concatenate(meshes);size=np.linalg.norm(mesh.bounds[1]-mesh.bounds[0])
    if not np.allclose(mesh.bounds,np.array([record['bbox_low'],record['bbox_high']]),atol=1e-5):
        raise ValueError('Blender and GLB bounds differ')
    origins=[];directions=[];expected=[];camera_z=[]
    for camera in record['views'][::5]:
        rgba=np.asarray(Image.open(folder/camera['rgba']));mask=(rgba[:,:,3]>254).astype(np.uint8)
        mask=cv2.erode(mask,np.ones((7,7),np.uint8));ys,xs=np.where(mask)
        chosen=np.linspace(0,len(xs)-1,18).astype(int);xs=xs[chosen];ys=ys[chosen]
        raw=read_depth_exr(folder/camera['depth'])
        rays=np.stack(((xs+.5-camera['cx'])/camera['fx'],(ys+.5-camera['cy'])/camera['fy'],np.ones(len(xs))),1)
        rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        R=np.asarray(camera['R']);center=-R.T@np.asarray(camera['t'])
        origins.extend(np.repeat(center[None],len(xs),axis=0));directions.extend(rays@R)
        expected.extend(raw[ys,xs]);camera_z.extend(rays[:,2])
    origins=np.asarray(origins);directions=np.asarray(directions);expected=np.asarray(expected)
    points,ids,_=mesh.ray.intersects_location(origins,directions,multiple_hits=False)
    measured=np.linalg.norm(points-origins[ids],axis=1);error=abs(measured-expected[ids])/size
    wrong_error=abs(measured*np.asarray(camera_z)[ids]-expected[ids])/size
    definition='camera_z' if np.median(wrong_error)<np.median(error) else 'ray_distance'
    chosen=wrong_error if definition=='camera_z' else error
    result={'rays':len(origins),'hits':len(ids),'object_diagonal':float(size),
        'distance_median_relative_error':float(np.median(error)),'distance_p95_relative_error':float(np.quantile(error,.95)),
        'camera_z_median_relative_error':float(np.median(wrong_error)),
        'glb_blender_bounds_match':True,'verified_definition':definition,
        'chosen_p95_relative_error':float(np.quantile(chosen,.95)),
        'passed':bool(len(ids)>.95*len(origins) and np.quantile(chosen,.95)<.005)}
    write_json(folder/'ray_depth_calibration.json',result)
    if not result['passed']: raise ValueError(str(result))
    record['depth_definition']=definition
    record['depth_definition_verification']='ray_depth_calibration.json: independent triangle intersections'
    write_json(folder/'cameras.json',record)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();validate(a.dataset)
