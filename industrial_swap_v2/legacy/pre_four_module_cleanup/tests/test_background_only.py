"""CPU regression tests for union exclusion and separated gradient permissions."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from industrial_swap_v2.core.multiview_masks import union_supervision_masks,occlusion_permissions,DEFAULTS
from industrial_swap_v2.core.local_background_optimizer import PermissionedBackground
from industrial_swap_v2.tests.test_multiview_refinement import Common


class UnionMaskTests(unittest.TestCase):
    def setUp(self):
        self.camera=SimpleNamespace(width=9,height=9,fx=9,fy=9,cx=4,cy=4,R=np.eye(3),center=np.zeros(3))
        self.box=(np.array([-.5,-.5,2]),np.array([.5,.5,3]))
        self.ones=np.ones((9,9)); self.zeros=np.zeros((9,9))

    def test_exact_union_excludes_both_objects_without_excluding_box(self):
        old=self.zeros.copy(); new=self.zeros.copy(); old[3,3]=1; new[4,5]=1
        mold,mnew,env,photo,legal,visible=union_supervision_masks(self.camera,self.box,
            self.ones*2.5,self.zeros,self.ones,old,self.ones*4,new,1.,DEFAULTS)
        np.testing.assert_array_equal(env,~(mold|mnew))
        self.assertFalse(photo[3,3]); self.assertFalse(photo[4,5])
        self.assertTrue(photo[4,4])  # Inside old box but no visible old/new surface.
        self.assertFalse(legal[4,5]); self.assertTrue(visible[4,5])

    def test_legal_foreground_is_independent_of_rgb_exclusion(self):
        _,_,env,_,legal,visible=union_supervision_masks(self.camera,self.box,
            self.ones,self.zeros,self.ones,self.zeros,self.ones*4,self.ones,1.,DEFAULTS)
        self.assertFalse(env[4,4]); self.assertTrue(legal[4,4]); self.assertFalse(visible[4,4])

    def test_occluders_without_photo_support_can_be_selected_but_need_spread(self):
        ids=np.array([9,14,27]); votes=np.array([[1,1,1],[1,1,1],[1,1,0]],bool)
        directions=np.array([[0,0,1],[.2,0,1],[-.2,0,1]],float)
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        selected,_=occlusion_permissions(ids,votes,directions,np.array([False,True,False]),DEFAULTS)
        self.assertEqual(selected.tolist(),[9])  # No RGB count input, legal veto honored.
        selected,_=occlusion_permissions(ids,votes,np.tile([0.,0.,1.],(3,1)),np.zeros(3,bool),DEFAULTS)
        self.assertEqual(selected.tolist(),[])


class PermissionTests(unittest.TestCase):
    def test_m4_background_only_returns_without_diffusion_import(self):
        from industrial_swap_v2.modules.m4_refinement import run
        ctx=SimpleNamespace(cfg={'m4':{'background_only':True},'m3':{'visibility_initialization':{'scheme':'old_new_union_v2'}}},stage=lambda _:Path('/unused'))
        with patch('industrial_swap_v2.core.local_background_optimizer.optimize_background') as optimize, \
             patch('industrial_swap_v2.scripts.run_background_only.validate_background_only') as validate:
            run(ctx)
            optimize.assert_called_once_with(ctx); validate.assert_called_once_with(ctx)

    def model(self):
        b=SimpleNamespace(xyz=torch.zeros((4,3)),log_scale=torch.zeros((4,3)),rotation=torch.ones((4,4)),
                          opacity_logit=torch.ones((4,1))*2,sh=torch.zeros((4,3,1)))
        cfg={'max_opacity_change':.08,'max_rgb_change':.08,'occlusion_min_alpha':.005}
        return PermissionedBackground(b,[0,1],[1,2],Common,cfg)

    def test_rgb_loss_cannot_update_occlusion_only_splats(self):
        model=self.model(); model.data('rgb').opacity_logit.sum().backward()
        np.testing.assert_array_equal(model.opacity_delta.grad.numpy().ravel(),[1,1,0])

    def test_visibility_loss_cannot_update_rgb_only_splats(self):
        model=self.model(); model.data('occlusion').opacity_logit.sum().backward()
        np.testing.assert_array_equal(model.opacity_delta.grad.numpy().ravel(),[0,1,1])

    def test_forward_values_agree_and_unselected_rows_are_exact(self):
        model=self.model()
        with torch.no_grad(): model.opacity_delta.fill_(-.3)
        a=model.data('rgb'); b=model.data('occlusion')
        torch.testing.assert_close(a.opacity_logit,b.opacity_logit,atol=0,rtol=0)
        self.assertEqual(a.opacity_logit[3].item(),2.)
        self.assertEqual(len(model.dc_delta),2)  # No color parameter for occlusion-only.

    def test_occlusion_opacity_only_decreases_and_rgb_change_is_bounded(self):
        model=self.model()
        with torch.no_grad(): model.opacity_delta.fill_(100)
        model.constrain(); alpha=model.data().opacity_logit.sigmoid()
        self.assertLessEqual(float(alpha[0]-model.initial_alpha[0]),.080001)
        self.assertEqual(float(model.opacity_delta[1]),0.); self.assertEqual(float(model.opacity_delta[2]),0.)
        with torch.no_grad(): model.opacity_delta.fill_(-100)
        model.constrain(); self.assertAlmostEqual(float(model.data().opacity_logit.sigmoid()[2]),.005,places=6)


if __name__=='__main__': unittest.main()
