"""CPU-only adapter integrity tests. Synthetic meshes are never experiment outputs."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
from industrial_swap_v2.core.mesh_to_gaussian import convert, SH_C0
from industrial_swap_v2.core.alignment import transform_vertices, xyz_of
from industrial_swap_v2.core.scene import compose, read_vertices, save_vertices

class GeometryTests(unittest.TestCase):
    def test_textured_mesh_color_covariance_and_lossless_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mesh=trimesh.creation.box()
            texture=Image.new('RGB',(4,4),(230,170,20))
            mesh.visual=trimesh.visual.TextureVisuals(uv=np.full((len(mesh.vertices),2),.5),
                material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,baseColorFactor=[255,255,255,255]))
            mesh.export(root/'test.glb')
            config={'gaussian_count':1000,'tangent_scale_multiplier':1.1,'normal_scale_ratio':.15,'opacity':.8}
            vertices=convert(root/'test.glb',root/'local.ply',config)
            color=np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])*SH_C0+.5
            np.testing.assert_allclose(color,np.tile(np.array([230,170,20])/255,(1000,1)),atol=1e-5)
            rot=Rotation.from_euler('xyz',[20,40,70],degrees=True).as_matrix()
            scale=2.3; translation=np.array([5.,-2.,3.])
            world=transform_vertices(vertices,scale,rot,translation)
            np.testing.assert_allclose(xyz_of(world),xyz_of(vertices)@rot.T*scale+translation,atol=1e-6)
            def cov(row):
                q=np.array([row[f'rot_{i}'] for i in range(4)])[[1,2,3,0]]
                r=Rotation.from_quat(q).as_matrix()
                return r@np.diag(np.exp([row[f'scale_{i}'] for i in range(3)])**2)@r.T
            np.testing.assert_allclose(cov(world[0]),scale**2*rot@cov(vertices[0])@rot.T,rtol=1e-5,atol=1e-7)
            save_vertices(root/'world.ply',world)
            save_vertices(root/'background.ply',vertices[:37])
            compose(root/'background.ply',root/'world.ply',root/'scene.ply',root/'composition.json')
            self.assertEqual(read_vertices(root/'scene.ply')[:37].tobytes(),vertices[:37].tobytes())

if __name__=='__main__': unittest.main()
