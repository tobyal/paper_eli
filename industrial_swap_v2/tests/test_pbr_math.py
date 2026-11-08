import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.pbr_math import sh_basis,sh_rotation_matrix,ray_distance_to_z


class PBRMathTests(unittest.TestCase):
    def test_directional_radiance_survives_object_rotation(self):
        rng=np.random.default_rng(7);directions=rng.normal(size=(80,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        rotation=Rotation.from_euler('xyz',[37,-58,19],degrees=True).as_matrix()
        coefficients=rng.normal(size=(3,16));matrix=sh_rotation_matrix(rotation)
        expected=sh_basis(directions@rotation)@coefficients.T
        actual=sh_basis(directions)@(coefficients@matrix.T).T
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(sh_rotation_matrix(rotation.T)@matrix,np.eye(16),atol=1e-12)

    def test_constant_z_plane_ray_distance_conversion(self):
        camera={'fx':400.,'fy':350.,'cx':31.5,'cy':20.5};y,x=np.indices((43,65))
        distance=2*np.sqrt(((x+.5-camera['cx'])/camera['fx'])**2+((y+.5-camera['cy'])/camera['fy'])**2+1)
        np.testing.assert_allclose(ray_distance_to_z(distance,camera),2,atol=1e-6)


if __name__=='__main__':unittest.main()
