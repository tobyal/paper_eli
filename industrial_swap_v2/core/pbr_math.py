"""Explicit GLB/CV calibration, depth conversion, and real-SH frame changes."""
import numpy as np


def read_depth_exr(path):
    """Cycles compositor BW files use a V channel, which OpenCV cannot decode."""
    import OpenEXR
    import Imath
    stream=OpenEXR.InputFile(str(path))
    try:
        header=stream.header();window=header['dataWindow'];channels=list(header['channels'])
        name=next((n for n in ('V','Z','Y','R') if n in channels),channels[0])
        width=window.max.x-window.min.x+1;height=window.max.y-window.min.y+1
        return np.frombuffer(stream.channel(name,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(height,width).copy()
    finally:
        stream.close()


def ray_distance_to_z(distance,camera):
    y,x=np.indices(distance.shape,dtype=np.float64)
    length=np.sqrt(((x+.5-camera['cx'])/camera['fx'])**2+
                   ((y+.5-camera['cy'])/camera['fy'])**2+1)
    return (distance/length).astype(np.float32)


def sh_basis(directions):
    """Graphdeco real SH order/signs through degree 3 (same as utils.sh_utils)."""
    x,y,z=np.asarray(directions,dtype=np.float64).T
    xx,yy,zz=x*x,y*y,z*z;xy,yz,xz=x*y,y*z,x*z
    return np.stack((np.full_like(x,.28209479177387814),
        -.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
        1.0925484305920792*xy,-1.0925484305920792*yz,
        .31539156525252005*(2*zz-xx-yy),-1.0925484305920792*xz,.5462742152960396*(xx-yy),
        -.5900435899266435*y*(3*xx-yy),2.890611442640554*xy*z,
        -.4570457994644658*y*(4*zz-xx-yy),.3731763325901154*z*(2*zz-3*xx-3*yy),
        -.4570457994644658*x*(4*zz-xx-yy),1.445305721320277*z*(xx-yy),
        -.5900435899266435*x*(xx-3*yy)),axis=-1)


def sh_rotation_matrix(rotation):
    """Exact bandwise SO(3) transform recovered from oversampled SH polynomials.

    world position = R @ local position. Thus f_world(d)=f_local(d @ R).
    Degree bands are solved independently; no color fit or view approximation.
    """
    rotation=np.asarray(rotation,dtype=np.float64)
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or np.linalg.det(rotation)<0:
        raise ValueError('SH rotation requires a proper rotation')
    i=np.arange(128);y=1-2*(i+.5)/len(i);phi=i*np.pi*(3-np.sqrt(5))
    dirs=np.stack((np.sqrt(1-y*y)*np.cos(phi),y,np.sqrt(1-y*y)*np.sin(phi)),1)
    target=sh_basis(dirs@rotation);source=sh_basis(dirs);matrix=np.zeros((16,16))
    for degree in range(4):
        sl=slice(degree**2,(degree+1)**2)
        matrix[sl,sl]=np.linalg.lstsq(source[:,sl],target[:,sl],rcond=None)[0]
    assert np.max(abs(source@matrix-target))<1e-10
    return matrix


def rotate_vertex_sh(vertices,rotation):
    names=[f'f_rest_{i}' for i in range(45)]
    if not all(name in vertices.dtype.names for name in names):
        raise ValueError('PBR integration expects degree-3 Graphdeco SH fields')
    if not any(np.any(vertices[name]) for name in names): return vertices
    transform=sh_rotation_matrix(rotation)[1:,1:]
    result=vertices.copy()
    for start in range(0,len(vertices),20000):
        end=min(start+20000,len(vertices))
        coefficients=np.stack([vertices[n][start:end] for n in names],axis=1).reshape(-1,3,15)
        rotated=coefficients@transform.T
        for index,name in enumerate(names): result[name][start:end]=rotated.reshape(-1,45)[:,index]
    return result
