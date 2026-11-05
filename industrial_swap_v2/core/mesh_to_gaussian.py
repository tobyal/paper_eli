"""Area-weighted textured mesh sampling into Graphdeco's log/SH convention."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from .scene import save_vertices
from .context import write_json

SH_C0 = 0.28209479177387814


def fields(rest=45):
    return (['x','y','z','nx','ny','nz'] + [f'f_dc_{i}' for i in range(3)]
            + [f'f_rest_{i}' for i in range(rest)] + ['opacity']
            + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])


def sample_mesh(mesh, count, rng):
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('Mesh has no finite surface area')
    faces = rng.choice(len(areas), count, p=areas/areas.sum())
    r = rng.random((count,2)); s = np.sqrt(r[:,0])
    bary = np.stack((1-s, s*(1-r[:,1]), s*r[:,1]),1)
    tri = np.asarray(mesh.triangles)[faces]
    xyz = np.einsum('ni,nij->nj',bary,tri)
    normal = np.asarray(mesh.face_normals)[faces]
    tangent = tri[:,1]-tri[:,0]
    tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
    bitangent = np.cross(normal,tangent)
    rotation = Rotation.from_matrix(np.stack((tangent,bitangent,normal),axis=2)).as_quat()[:,[3,0,1,2]]
    visual = mesh.visual
    if visual.kind == 'texture':
        uv = np.einsum('ni,nij->nj',bary,np.asarray(visual.uv)[mesh.faces[faces]])
        material = visual.material
        texture = getattr(material,'baseColorTexture',None)
        if texture is None:
            texture = getattr(material,'image',None)
        factor = getattr(material,'baseColorFactor',None)
        factor = np.asarray(factor if factor is not None else [255,255,255,255],float)[:3]/255
        if texture is not None:
            rgb = trimesh.visual.color.uv_to_color(uv,texture)[:,:3]/255*factor
        else:
            rgb = np.broadcast_to(factor,(count,3)).copy()
    elif visual.kind == 'vertex':
        rgb = np.einsum('ni,nij->nj',bary,np.asarray(visual.vertex_colors)[mesh.faces[faces],:3])/255
    else:
        rgb = np.asarray(visual.face_colors)[faces,:3]/255
    return xyz, normal, rotation, np.clip(rgb,0,1)


def convert(glb_path, destination, config, seed=42):
    scene = trimesh.load(str(glb_path),force='scene',process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        if len(mesh.faces) and mesh.area > 0:
            meshes.append(mesh)
    if not meshes:
        raise ValueError('GLB contains no triangle meshes')
    total_area = sum(m.area for m in meshes)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(config['gaussian_count'],[m.area/total_area for m in meshes])
    samples = [sample_mesh(m,int(n),rng) for m,n in zip(meshes,counts) if n]
    xyz, normal, rotation, rgb = [np.concatenate([s[i] for s in samples]) for i in range(4)]
    n = len(xyz)
    # Ellipsoids lie in the triangle tangent plane; thickness is a configurable
    # fraction of typical area-per-sample spacing, not an extra geometry model.
    spacing = np.sqrt(total_area/n) * config['tangent_scale_multiplier']
    scales = np.broadcast_to([spacing,spacing,spacing*config['normal_scale_ratio']],(n,3))
    vertices = np.zeros(n,dtype=[(k,'f4') for k in fields()])
    for names,values in ((['x','y','z'],xyz),(['nx','ny','nz'],normal),
                         ([f'rot_{i}' for i in range(4)],rotation),
                         ([f'scale_{i}' for i in range(3)],np.log(scales)),
                         ([f'f_dc_{i}' for i in range(3)],(rgb-.5)/SH_C0)):
        for i,key in enumerate(names): vertices[key]=values[:,i]
    opacity = config['opacity']
    if not 0 < opacity < 1: raise ValueError('Opacity must be in (0,1)')
    vertices['opacity'] = np.log(opacity/(1-opacity))
    save_vertices(Path(destination),vertices)
    write_json(Path(destination).with_suffix('.json'),{
        'mesh':str(glb_path),'points':n,'surface_area':float(total_area),
        'seed':seed,'method':'area weighted triangle surface sampling, barycentric UV color',
        'coordinate_system':'official GLB: Y up; native TRELLIS (x,y,z) -> GLB (x,z,-y)',
        'material_limitation':'Gaussian SH0 stores base color; metallic/roughness and normal maps remain in original GLB',
        'tangent_sigma':float(spacing),'normal_sigma':float(scales[0,2])})
    return vertices
