"""Local three-dimensional background repair, leaving M1 and object_world immutable.

Uses visibility attribution to prune abnormal Gaussian supports and interpolates
a thin floor surface from nearby reliable floor samples. No generation models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from .core.context import Context, gsle, read_json, write_json, sha256
from .core.scene import read_vertices, save_vertices, compose


def polygon_distance(points, polygon):
    equations=ConvexHull(polygon).equations
    return np.max(points@equations[:,:2].T+equations[:,2],axis=1)


def robust_floor(local, eligible, rng):
    ids=np.flatnonzero(eligible)
    if len(ids)<35: raise ValueError('Insufficient reliable neighboring floor samples')
    pts=local[ids]; design=np.column_stack((pts[:,0],pts[:,2],np.ones(len(pts))))
    best=None
    for _ in range(500):
        sample=rng.choice(len(pts),3,replace=False)
        try: coef=np.linalg.solve(design[sample],pts[sample,1])
        except np.linalg.LinAlgError: continue
        if np.linalg.norm(coef[:2])>.2: continue
        inlier=np.abs(design@coef-pts[:,1])<.014
        if best is None or inlier.sum()>best.sum(): best=inlier
    if best is None or best.sum()<25: raise ValueError('No supported local floor plane')
    coef=np.linalg.lstsq(design[best],pts[best,1],rcond=None)[0]
    return coef,ids[best]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--inspection',required=True)
    p.add_argument('--output',required=True); p.add_argument('--spacing',type=float,default=.006)
    p.add_argument('--contact-width',type=float,default=.30); p.add_argument('--planar',action='store_true')
    p.add_argument('--patch-margin',type=float,default=.16)
    p.add_argument('--floor-offset',type=float,default=-.012,
                   help='Keep the completed surface below retained thin floor/paint splats')
    args=p.parse_args(); ctx=Context(args.config); dest=ctx.path(args.output); dest.mkdir(parents=True,exist_ok=False)
    inspection=ctx.path(args.inspection); stats=np.load(inspection/'attributes.npz'); defect=np.load(inspection/'dark_attribution.npz')
    c=gsle(ctx); m=read_json(ctx.stage('m1')/'manifest.json'); camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    axis=stats['axes']; center=stats['center']; local=stats['local']; scale=stats['scales']; rgb=stats['colors']
    maximum=scale.max(1); normal_alignment=stats['normal_alignment']; opacity=stats['opacity']
    footprint=defect['footprint']; poly_dist=polygon_distance(local[:,[0,2]],footprint)
    floor0=float(defect['plane_y']); rng=np.random.default_rng(42)
    source_bg=ctx.stage('m1')/'background_gaussians.ply'; source_obj=ctx.stage('m3')/'object_world.ply'
    bg_vertices=read_vertices(source_bg); obj_vertices=read_vertices(source_obj)
    initial_hashes={'background':sha256(source_bg),'object':sha256(source_obj),'transform':sha256(ctx.stage('m3')/'transform.json'),
        'm1_manifest':sha256(ctx.stage('m1')/'manifest.json')}
    yellow=(rgb[:,0]>.25)&(rgb[:,1]>.18)&(rgb[:,2]<.55*np.minimum(rgb[:,0],rgb[:,1]))&(rgb[:,0]<1.8*rgb[:,1])
    neutral=(rgb.max(1)-rgb.min(1)<.19)&(rgb.mean(1)>.23)&(rgb.mean(1)<.8)
    donors=(poly_dist>.03)&(poly_dist<.85)&(abs(local[:,1]-floor0)<.09)&(normal_alignment>.93)&(maximum<.065)&(opacity>.2)&neutral&(stats['floor_scores']>.43)&~yellow
    plane,donor_ids=robust_floor(local,donors,rng)
    plane_y=local[:,0]*plane[0]+local[:,2]*plane[1]+plane[2]
    object_xyz=np.column_stack([obj_vertices[k] for k in ('x','y','z')]); object_local=(object_xyz-center)@axis
    bottom=object_local[:,1].min(); base=object_local[:,1]<bottom+.15*np.ptp(object_local[:,1])
    base_points=object_local[base][:,[0,2]]; base_poly=base_points[ConvexHull(base_points).vertices]
    def height(points):
        value=points[:,0]*plane[0]+points[:,1]*plane[1]+plane[2]+args.floor_offset
        if not args.planar:
            distance=np.maximum(polygon_distance(points,base_poly)-.025,0)
            contact=np.exp(-(distance/args.contact_width)**4)
            edge=np.clip((args.patch_margin-polygon_distance(points,footprint))/.20,0,1)
            blend=contact*(edge*edge*(3-2*edge))
            # Geometry completion under the immutable device: surface meets
            # its support, then smoothly joins the measured neighboring floor.
            value-=np.maximum(value-(bottom-.003),0)*blend
        return value
    # Only low background supports reaching the old-device footprint are eligible.
    quats=np.column_stack([bg_vertices[f'rot_{i}'] for i in range(4)])[:,[1,2,3,0]]
    rotation=Rotation.from_quat(quats).as_matrix()
    sigma_up=np.sqrt(np.sum(((rotation.transpose(0,2,1)@axis[:,1])*scale)**2,axis=1))
    support=(poly_dist<np.minimum(3*maximum,.95)+.025)&(abs(local[:,1]-plane_y)<3*sigma_up+.16)&(local[:,1]<floor0+.24)
    malformed=(normal_alignment<.92)|(maximum>.05)|(abs(local[:,1]-plane_y)>.04)|((rgb.mean(1)<.23)&(maximum>.018))
    dark_bad=(defect['score']>.1)&malformed
    base_occluder=stats['occlusion_score']>.002
    # Account for local floor completion beneath the fixed base; preserve all
    # retained original ground outside this tightly bounded contact area.
    contact_conflict=(polygon_distance(local[:,[0,2]],base_poly)<.08)&(local[:,1]>height(local[:,[0,2]])+.012)&(abs(local[:,1]-plane_y)<.14)
    remove=support&(dark_bad|base_occluder|contact_conflict)&~yellow
    if remove.sum()>len(bg_vertices)*.01: raise ValueError('Repair would remove more than 1% of background')
    kept=bg_vertices[~remove].copy()
    np.save(dest/'removed_background_indices.npy',np.flatnonzero(remove))
    np.save(dest/'retained_background_indices.npy',np.flatnonzero(~remove))
    np.save(dest/'protected_yellow_indices.npy',np.flatnonzero(yellow))
    np.save(dest/'floor_donor_indices.npy',donor_ids)
    save_vertices(dest/'removed_gaussians.ply',bg_vertices[remove])
    save_vertices(dest/'background_cleaned.ply',kept)
    # Regular coverage in the floor tangent coordinates, not image-space pixels.
    lo=footprint.min(0)-args.patch_margin; hi=footprint.max(0)+args.patch_margin; step=args.spacing
    xx,zz=np.meshgrid(np.arange(lo[0],hi[0]+step,step),np.arange(lo[1],hi[1]+step,step))
    coords=np.column_stack((xx.ravel(),zz.ravel())); pd=polygon_distance(coords,footprint)
    coords=coords[pd<args.patch_margin-.001]; pd=pd[pd<args.patch_margin-.001]
    yy=height(coords); world=np.column_stack((coords[:,0],yy,coords[:,1]))@axis.T+center
    # Color a three-dimensional surface using *composited* observed floor,
    # rather than individual SH coefficients (which are not surface albedo).
    bg=c.GaussianData.load(source_bg)
    with torch.no_grad():
        observed=c.render(camera,bg,aux=True)
        # Radiance contributors are less geometrically precise than the
        # RANSAC plane donors: retain semantic floor contributors with broad
        # anisotropic support, while checking rendered depth and color too.
        floor_flags=(abs(local[:,1]-plane_y)<.16)&(normal_alignment>.60)&(stats['floor_scores']>.42)
        floor_coverage=c.raster(camera,bg,torch.tensor(floor_flags,device='cuda',dtype=torch.float32)[:,None].expand(-1,3))
    sx,sz=np.meshgrid(np.arange(lo[0]-.65,hi[0]+.65,.012),np.arange(lo[1]-.65,hi[1]+.65,.012))
    samples=np.column_stack((sx.ravel(),sz.ravel()))
    sample_y=samples[:,0]*plane[0]+samples[:,1]*plane[1]+plane[2]
    sample_world=np.column_stack((samples[:,0],sample_y,samples[:,1]))@axis.T+center
    uv,z=camera.project(sample_world); u=uv[:,0].astype(np.float32).reshape(-1,1); v=uv[:,1].astype(np.float32).reshape(-1,1)
    def at(tensor):
        arr=tensor.detach().cpu().numpy()
        if arr.ndim==3: arr=arr.transpose(1,2,0)
        # OpenCV uses signed 16-bit map dimensions internally.
        return np.concatenate([cv2.remap(arr,u[i:i+16000],v[i:i+16000],cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT).reshape(min(16000,len(samples)-i),-1)
            for i in range(0,len(samples),16000)])
    observed_colors=at(observed['rgb']); observed_depth=at(observed['depth'])[:,0]
    coverage=at(floor_coverage)[:,0]
    expected_depth=np.linalg.norm(sample_world-camera.center,axis=1)
    sd=polygon_distance(samples,footprint)
    color_valid=(sd>.04)&(sd<.7)&(coverage>.35)&(abs(observed_depth-expected_depth)<.25)&(z>0)
    color_valid&=(observed_colors.max(1)-observed_colors.min(1)<.18)&(observed_colors.mean(1)>.30)&(observed_colors.mean(1)<.82)
    if color_valid.sum()<100: raise ValueError('Insufficient visible neighboring floor texture samples')
    color_coords=samples[color_valid]; color_rgb=observed_colors[color_valid]
    np.savez_compressed(dest/'floor_color_samples.npz',local_xz=color_coords,rgb=color_rgb)
    tree=cKDTree(color_coords); distances,nearest=tree.query(coords,k=min(48,len(color_coords)))
    weights=1/np.maximum(distances,.03)**2; weights/=weights.sum(1,keepdims=True)
    colors=np.einsum('nk,nkc->nc',weights,color_rgb[nearest])
    dc=(colors-.5)/.28209479177387814
    patch=np.zeros(len(coords),dtype=bg_vertices.dtype)
    for i,name in enumerate(('x','y','z')): patch[name]=world[:,i]
    for i in range(3): patch[f'f_dc_{i}']=dc[:,i]
    # A diffuse floor completion: no invented view-dependent texture.
    eps=.001
    gx=(height(coords+[eps,0])-height(coords-[eps,0]))/(2*eps)
    gz=(height(coords+[0,eps])-height(coords-[0,eps]))/(2*eps)
    tangent=np.column_stack((np.ones(len(coords)),gx,np.zeros(len(coords))))@axis.T
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    normal=np.column_stack((-gx,np.ones(len(coords)),-gz))@axis.T; normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    bitangent=np.cross(normal,tangent)
    quat=Rotation.from_matrix(np.stack((tangent,bitangent,normal),axis=2)).as_quat()[:,[3,0,1,2]]
    for i in range(4): patch[f'rot_{i}']=quat[:,i]
    for i,name in enumerate(('nx','ny','nz')): patch[name]=normal[:,i]
    # Account for the surface Jacobian so steep transition regions have the
    # same coverage as flat parts instead of exposing stripes between rows.
    patch['scale_0']=np.log(step*.90*np.sqrt(1+gx*gx))
    patch['scale_1']=np.log(step*.90*np.sqrt(1+gz*gz/(1+gx*gx)))
    patch['scale_2']=np.log(step*.06)
    fade=np.clip((args.patch_margin-pd)/.12,.001,1)
    alpha=.95*fade*fade*(3-2*fade); patch['opacity']=np.log(alpha/(1-alpha))
    save_vertices(dest/'floor_patch.ply',patch)
    fixed=np.concatenate((kept,patch)); save_vertices(dest/'background_fixed.ply',fixed)
    # Explicit byte-level invariants, including every retained original Gaussian.
    assert kept.tobytes()==bg_vertices[~remove].tobytes()
    assert not np.any(remove&yellow)
    compose(dest/'background_fixed.ply',source_obj,dest/'scene_coarse_fixed.ply',dest/'composition.json')
    complete=read_vertices(dest/'scene_coarse_fixed.ply')
    assert complete[-len(obj_vertices):].tobytes()==obj_vertices.tobytes()
    assert sha256(source_obj)==initial_hashes['object'] and sha256(source_bg)==initial_hashes['background']
    obj=c.GaussianData.load(source_obj)
    repaired=c.GaussianData.load(dest/'background_fixed.ply'); cleaned=c.GaussianData.load(dest/'background_cleaned.ply')
    views=[camera,*[v for v in c.source_cameras(938) if v.name in ('frame_000101.jpg','frame_000151.jpg','frame_000201.jpg')]]
    renders=dest/'renders'; renders.mkdir()
    records=[]
    with torch.no_grad():
        for cam in views:
            name=Path(cam.name).stem
            before=c.render_joint(cam,bg,obj,aux=False)['rgb']; after=c.render_joint(cam,repaired,obj,aux=False)['rgb']
            c.save_image(renders/(name+'_before.png'),before); c.save_image(renders/(name+'_after.png'),after)
            c.save_image(renders/(name+'_before_after.png'),torch.cat((before,after),dim=2))
            records.append(cam.record())
        c.save_image(renders/'background_before.png',c.render(camera,bg,aux=False)['rgb'])
        c.save_image(renders/'background_cleaned.png',c.render(camera,cleaned,aux=False)['rgb'])
        c.save_image(renders/'background_fixed.png',c.render(camera,repaired,aux=False)['rgb'])
        flags=torch.tensor(base,device='cuda',dtype=torch.float32)[:,None].expand(-1,3)
        isolated=c.raster(camera,obj,flags)[0].sum(); visibility={}
        for name,source in [('before',bg),('after',repaired)]:
            joint=c.concatenate(source,obj)
            mask=c.raster(camera,joint,torch.cat((torch.zeros((len(source.xyz),3),device='cuda'),flags)))[0]
            visibility[name]=float(mask.sum()/isolated); c.save_image(renders/(name+'_base_visibility.png'),mask)
    report={'status':'rendered_for_review','source_config':str(ctx.config_path),'source_hashes':initial_hashes,
        'original_background_count':len(bg_vertices),'removed_count':int(remove.sum()),'patch_count':len(patch),
        'retained_original_count':len(kept),'object_count':len(obj_vertices),'object_bitwise_unchanged':True,
        'retained_background_bitwise_unchanged':True,'yellow_gaussians_removed':0,
        'local_plane_y_ax_bz_c':plane.tolist(),'plane_donor_count':len(donor_ids),
        'floor_color_sample_count':int(color_valid.sum()),
        'patch_height_range':[float(yy.min()),float(yy.max())],'existing_object_bottom_local_y':float(bottom),
        'contact_completion':not args.planar,'contact_width':args.contact_width,
        'patch_margin':args.patch_margin,
        'floor_offset':args.floor_offset,'inspection_path':str(inspection),
        'patch_spacing':step,'repair_footprint':footprint.tolist(),'base_visibility_ratio':visibility,
        'no_qwen_or_trellis_calls':True,'cameras':records,
        'method':'3D support + actual occlusion/dark-region contribution pruning; local robust floor geometry and depth/semantic-checked neighboring floor radiance interpolation; fixed-object contact constraint'}
    write_json(dest/'repair_report.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
