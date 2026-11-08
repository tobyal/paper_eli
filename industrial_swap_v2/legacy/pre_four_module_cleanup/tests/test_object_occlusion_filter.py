"""Geometry and conservative selection regression tests, independent of CUDA."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.object_occlusion_filter import filter_config, spatial_candidates, select_deletions
from industrial_swap_v2.core.context import sha256
from industrial_swap_v2.core.m3_composition import effective_background


class OcclusionTests(unittest.TestCase):
    def fixture(self):
        names=['x','y','z']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        v=np.zeros(3,dtype=[(n,'f4') for n in names]); v['x']=[0,1,1]
        for n in ('scale_0','scale_1','scale_2'): v[n]=np.log(.01)
        v['scale_0'][1]=np.log(.3); v['scale_1'][2]=np.log(.3); v['rot_0']=1
        region={'center':[0,0,0],'axes':np.eye(3).tolist(),'local_bbox_low':[-.5]*3,'local_bbox_high':[.5]*3}
        return v,region

    def test_covariance_reaches_device_from_outside_edit_center(self):
        v,region=self.fixture(); cfg=filter_config({'enabled':True})
        ids,*_=spatial_candidates(v,np.zeros((1,3)),region,cfg)
        self.assertEqual(ids.tolist(),[0,1])
        # Rotate the long Y ellipsoid into X: its center is still outside.
        q=Rotation.from_euler('z',90,degrees=True).as_quat()[[3,0,1,2]]
        for i in range(4): v[f'rot_{i}'][2]=q[i]
        ids,*_=spatial_candidates(v,np.zeros((1,3)),region,cfg)
        self.assertEqual(ids.tolist(),[0,1,2])

    def test_relative_spatial_constraints_are_scale_translation_invariant(self):
        v,region=self.fixture(); cfg=filter_config({'enabled':True})
        before=spatial_candidates(v,np.zeros((1,3)),region,cfg)[0]
        shift=np.array([12.,-8.,32.]); factor=23.
        for i,key in enumerate(('x','y','z')): v[key]=v[key]*factor+shift[i]
        for i in range(3): v[f'scale_{i}']+=np.log(factor)
        region['center']=shift.tolist(); region['local_bbox_low']=(np.array(region['local_bbox_low'])*factor).tolist()
        region['local_bbox_high']=(np.array(region['local_bbox_high'])*factor).tolist()
        after=spatial_candidates(v,shift[None],region,cfg)[0]
        np.testing.assert_array_equal(before,after)

    def test_multiview_and_outside_budget_retain_legitimate_or_uncertain_occluders(self):
        cfg=filter_config({'enabled':True})
        gain=np.array([[20,20,20,20],[20,20,20,0.]])
        inside=np.full((2,4),100.); outside=np.array([[0,2000,30,0],[0,2000,30,0.]])
        chosen,votes,_=select_deletions(gain,inside,outside,np.ones((2,4),bool),
            np.array([10000,10000]),np.array([10000,10000]),cfg,10000)
        self.assertEqual(chosen.tolist(),[0])
        self.assertEqual(votes[:,1].sum(),0)  # Broad real foreground.
        self.assertEqual(votes[:,3].sum(),1)  # Insufficient view confirmation.

    def test_no_color_or_floor_options_and_strict_safety_limits(self):
        for settings in ({'floor_repair':True},{'preserve_object_geometry':False},{'yellow_protection':True},
                         {'max_delete_fraction':.1},{'min_occlusion_fraction':float('nan')},{'enabled':'yes'}):
            with self.subTest(settings=settings),self.assertRaises(ValueError): filter_config(settings)

    def test_downstream_background_resolution_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'m1').mkdir(); (root/'m3/occlusion_filter').mkdir(parents=True)
            class Ctx:
                def stage(self,name): return root/name
            ctx=Ctx(); old=root/'m1/background_gaussians.ply'; old.write_bytes(b'original')
            self.assertEqual(effective_background(ctx),old)
            filtered=root/'m3/occlusion_filter/background_filtered.ply'; filtered.write_bytes(b'filtered')
            selection=root/'m3/background_selection.json'
            selection.write_text(json.dumps({'filter_enabled':True,'relative_path':'occlusion_filter/background_filtered.ply','sha256':sha256(filtered)}))
            self.assertEqual(effective_background(ctx),filtered)
            filtered.write_bytes(b'changed')
            with self.assertRaises(ValueError): effective_background(ctx)
            selection.write_text(json.dumps({'filter_enabled':True,'relative_path':'../m1/background_gaussians.ply','sha256':sha256(old)}))
            with self.assertRaises(ValueError): effective_background(ctx)


if __name__=='__main__': unittest.main()
