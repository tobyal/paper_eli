"""Regression checks for supervision contamination, freezing, and opacity gradients."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from industrial_swap_v2.core.multiview_masks import ray_box_depth, supervision_masks, local_support, DEFAULTS as MASK_DEFAULTS
from industrial_swap_v2.core.local_background_optimizer import LocalBackground, DEFAULTS
from industrial_swap_v2.core.visibility_loss import masked_l1, visibility_loss
from industrial_swap_v2.core.m3_composition import effective_background
from industrial_swap_v2.core.context import sha256


class Common:
    @staticmethod
    def GaussianData(xyz,log_scale,rotation,opacity_logit,sh):
        return SimpleNamespace(xyz=xyz,log_scale=log_scale,rotation=rotation,opacity_logit=opacity_logit,sh=sh)
    @staticmethod
    def concatenate(bg,obj):
        return SimpleNamespace(opacity_logit=torch.cat((bg.opacity_logit,obj.opacity_logit)))
    @staticmethod
    def raster(camera,data,colors):
        alpha=data.opacity_logit.sigmoid()[:,0]
        trans=torch.cumprod(torch.cat((torch.ones(1),1-alpha[:-1])),0)
        return ((alpha*trans)[:,None]*colors).sum(0)[:,None,None]


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.camera=SimpleNamespace(width=9,height=9,fx=9,fy=9,cx=4,cy=4,R=np.eye(3),center=np.zeros(3))
        self.box=(np.array([-.5,-.5,2]),np.array([.5,.5,3]))
    def masks(self,z,old=0):
        ones=np.ones((9,9)); zeros=ones*0
        return supervision_masks(self.camera,self.box,ones*z,zeros,ones,ones*old,ones*4,ones,ones,1.,MASK_DEFAULTS)
    def test_camera_z_ray_interval(self):
        hit,near,far=ray_box_depth(self.camera,*self.box)
        self.assertTrue(hit[4,4]); self.assertEqual(near[4,4],2); self.assertEqual(far[4,4],3)
        self.assertFalse(hit[0,0])
    def test_old_equipment_pixels_are_not_background_targets(self):
        reliable,visible,excluded,legal,_=self.masks(2.5)
        self.assertFalse(reliable[4,4]); self.assertTrue(excluded[4,4]); self.assertFalse(legal[4,4])
        # The visibility constraint is distinct from real-image supervision.
        self.assertTrue(visible[4,4])
    def test_trusted_foreground_is_not_forced_transparent(self):
        reliable,visible,_,legal,_=self.masks(1.)
        self.assertTrue(reliable[4,4]); self.assertTrue(legal[4,4]); self.assertFalse(visible[4,4])
    def test_background_seen_beyond_old_box_is_usable(self):
        reliable,visible,excluded,_,_=self.masks(5.)
        self.assertTrue(reliable[4,4]); self.assertFalse(excluded[4,4]); self.assertTrue(visible[4,4])
    def test_old_semantic_contribution_overrides_depth(self):
        reliable,_,excluded,_,_=self.masks(5.,.5)
        self.assertFalse(reliable[4,4]); self.assertTrue(excluded[4,4])
    def test_covariance_support_and_coordinate_scale(self):
        names=['x','y','z']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        v=np.zeros(3,dtype=[(n,'f4') for n in names]); v['x']=[0,1,4]; v['rot_0']=1
        for name in ('scale_0','scale_1','scale_2'): v[name]=np.log(.01)
        v['scale_0'][1]=np.log(.3)
        frame={'center':[0,0,0],'axes':np.eye(3).tolist(),'local_bbox_low':[-.5]*3,'local_bbox_high':[.5]*3}
        ids,_=local_support(v,{'world_transform':frame},0); self.assertEqual(ids.tolist(),[0,1])
        for name in ('x','y','z'): v[name]=v[name]*20+11
        for name in ('scale_0','scale_1','scale_2'): v[name]+=np.log(20)
        frame.update(center=[11]*3,local_bbox_low=[-10]*3,local_bbox_high=[10]*3)
        transformed,_=local_support(v,{'world_transform':frame},0); np.testing.assert_array_equal(ids,transformed)


class OptimizationTests(unittest.TestCase):
    def data(self,alpha):
        count=len(alpha)
        return Common.GaussianData(torch.zeros(count,3),torch.zeros(count,3),torch.zeros(count,4),
            torch.logit(torch.tensor(alpha))[:,None],torch.zeros(count,3,16))
    def test_exact_zero_initialization_and_only_selected_rows_change(self):
        original=self.data([.2,.8,.6]); model=LocalBackground(original,np.array([1]),np.array([1]),Common,DEFAULTS)
        self.assertTrue(torch.equal(model.data().opacity_logit,original.opacity_logit))
        with torch.no_grad(): model.opacity_delta.add_(.5)
        updated=model.data()
        self.assertTrue(torch.equal(updated.opacity_logit[[0,2]],original.opacity_logit[[0,2]]))
        self.assertTrue(torch.equal(updated.xyz,original.xyz)); self.assertTrue(torch.equal(updated.sh,original.sh))
    def test_non_candidate_opacity_change_is_bounded(self):
        original=self.data([.8,.8]); model=LocalBackground(original,np.array([0,1]),np.array([1]),Common,DEFAULTS)
        with torch.no_grad(): model.opacity_delta.fill_(-100)
        model.constrain(); alpha=model.data().opacity_logit.sigmoid()[:,0]
        self.assertAlmostEqual(float(alpha[0]),.68,places=5); self.assertAlmostEqual(float(alpha[1]),.2,places=5)
    def test_actual_front_to_back_visibility_gradient(self):
        bg=self.data([.5]); obj=self.data([.8]); bg.opacity_logit.requires_grad_(True)
        loss,visible=visibility_loss(Common,None,bg,obj,torch.tensor([[[.8]]]),torch.ones(1,1))
        self.assertAlmostEqual(float(visible),.4,places=6); loss.backward()
        self.assertGreater(float(bg.opacity_logit.grad),0)
    def test_legal_foreground_zero_weight_has_no_visibility_gradient(self):
        bg=self.data([.5]); obj=self.data([.8]); bg.opacity_logit.requires_grad_(True)
        loss,_=visibility_loss(Common,None,bg,obj,torch.tensor([[[.8]]]),torch.zeros(1,1)); loss.backward()
        self.assertEqual(float(bg.opacity_logit.grad),0)
    def test_masked_rgb_loss_cannot_learn_old_pixels(self):
        rgb=torch.zeros(3,2,2,requires_grad=True); truth=torch.ones(3,2,2); mask=torch.zeros(2,2); mask[0,0]=1
        loss=masked_l1(rgb,truth,mask); self.assertEqual(float(loss),1); loss.backward()
        self.assertTrue(torch.equal(rgb.grad[:,1],torch.zeros(3,2)))
    def test_m4_background_never_falls_back_to_m1(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            class Context:
                cfg={'m4':{'local_background':{'enabled':True}}}
                def stage(self,name): return root/name
            ctx=Context(); folder=root/'m4/background_optimization'; folder.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError): effective_background(ctx,'m4')
            path=folder/'background_optimized.ply'; path.write_bytes(b'optimized')
            (folder/'summary.json').write_text(json.dumps({'background_sha256':sha256(path)}))
            self.assertEqual(effective_background(ctx,'m4'),path)
            path.write_bytes(b'mutated')
            with self.assertRaises(ValueError): effective_background(ctx,'m4')


if __name__=='__main__': unittest.main(verbosity=2)
