import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.gaussian_registration import Pose,DEFAULTS,yaw_matrix,quaternion_product,pose_report


class RegistrationGeometry(unittest.TestCase):
    def test_quaternion_and_yaw_agree_with_scipy(self):
        for degrees in (-179,-30,0,81,180):
            r=Rotation.from_euler('xyz',[21,-12,36],degrees=True)
            a=torch.tensor(r.as_quat()[[3,0,1,2]])
            y=torch.tensor(np.deg2rad(degrees));zero=y*0
            q=quaternion_product(a,torch.stack(((y/2).cos(),zero,(y/2).sin(),zero)))
            actual=Rotation.from_quat(q.numpy()[[1,2,3,0]]).as_matrix()
            expected=(r*Rotation.from_euler('y',degrees,degrees=True)).as_matrix()
            np.testing.assert_allclose(actual,expected,atol=1e-7)
            np.testing.assert_allclose(yaw_matrix(y),Rotation.from_euler('y',degrees,degrees=True).as_matrix(),atol=1e-7)

    def test_support_is_fixed_under_scale_yaw_translation(self):
        rng=np.random.default_rng(5);xyz=torch.tensor(rng.uniform(-.5,.5,(300,3)),dtype=torch.float32)
        data=SimpleNamespace(xyz=xyz,log_scale=torch.full((300,3),-6.),opacity_logit=torch.ones(300,1),
            rotation=torch.tensor([[1.,0,0,0]]).repeat(300,1))
        axes=Rotation.from_euler('xyz',[155,10,20],degrees=True).as_matrix()
        frame={'axes':axes.tolist(),'center':[4,-2,3],'local_bbox_low':[-1,-.5,-1],'local_bbox_high':[1,.5,1]}
        model=Pose(None,data,{'frame':frame,'support_center_local':[.1,-.5,.2]},DEFAULTS)
        for yaw in (-2.,.7,3.):
            p=torch.tensor([yaw,np.log(.8),.1,-.2],requires_grad=True)
            # Match the production model dtype.
            p=p.float();scale,rot,t,_,_=model.placement(p)
            support=t+rot@model.pivot*scale
            position=(support-model.center)@model.axes
            self.assertAlmostEqual(float(position[1]),-.5,places=5)
            self.assertAlmostEqual(float(position[0]),.3,places=5)
            self.assertAlmostEqual(float(position[2]),-.2,places=5)
            (support.square().sum()).backward()
            report=pose_report(model,p.detach())
            rotation=np.asarray(report['rotation'])
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
            from industrial_swap_v2.core.pbr_math import sh_rotation_matrix
            sh_rotation_matrix(rotation)  # Regression: float32 rotations failed the exact-band check.
        excessive=torch.tensor([.8,np.log(model.base_scale*1.45),.3,.3],dtype=torch.float32)
        model.project_volume(excessive)
        self.assertLessEqual(pose_report(model,excessive)['max_volume_overflow_fraction'],DEFAULTS['max_volume_fraction'])


if __name__=='__main__':unittest.main()
