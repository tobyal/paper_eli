"""Geometry/mask invariants; no API, model generation, or optimization."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from industrial_swap_v2.core.m2_space import (DEFAULTS, VERSION, build_generation_region,
    semantic_profile, signature, generation_frame)
from industrial_swap_v2.core.m2_edit_mask import build_mask, ray_box
from industrial_swap_v2.core.context import write_json, sha256


def fixture():
    rng=np.random.default_rng(8); n=1500
    v=np.zeros(n,dtype=[(k,'f4') for k in ('x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3')])
    xyz=rng.uniform([-1,0,-.6],[1,1,.6],(n,3))
    for i,k in enumerate(('x','y','z')):v[k]=xyz[:,i]
    v['opacity']=3; v['rot_0']=1
    for i in range(3):v[f'scale_{i}']=np.log(.01*(i+1))
    target={'text':'old device','up':[0,1,0], 'reference_camera':{'R':np.eye(3).tolist(),'t':[0,-1,4]},
            'up_estimation':{'normal':[0,1,0],'offset':0}}
    return v,target


class Camera:
    width=height=100; fx=fy=100.; cx=cy=50.; R=np.eye(3); t=np.zeros(3); center=np.zeros(3)
    def project(self,p):
        p=np.asarray(p); return p[:,:2]/p[:,2:]*100+50,p[:,2]


class SpatialTests(unittest.TestCase):
    def test_semantics_unknown_and_compound_description(self):
        self.assertEqual(semantic_profile('A white inspection station with a lower cabinet')['name'],'gantry_station')
        self.assertEqual(semantic_profile('A purple unfamiliar processing device')['name'],'geometry_fallback')
        v,t=fixture()
        a=build_generation_region(v,t,'A yellow robot arm with a base',DEFAULTS)
        b=build_generation_region(v,t,'A roller conveyor with legs',DEFAULTS)
        self.assertGreater(a['geometry']['new_side_lengths'][1],b['geometry']['new_side_lengths'][1])
        self.assertAlmostEqual(a['contact']['floor_local_y'],b['contact']['floor_local_y'])
        self.assertTrue(a['contact']['snapped'])
        cfg={'project':{'source_sha256':'same'},'m2':{'replacement_text':'A robot arm'}}
        other=copy.deepcopy(cfg);other['m2']['replacement_text']='A conveyor'
        self.assertNotEqual(signature(cfg),signature(other))

    def test_rigid_and_scale_equivariance(self):
        v,t=fixture(); original=v.tobytes()
        a=build_generation_region(v,t,'A robot arm',DEFAULTS)
        r=Rotation.from_euler('xyz',[19,48,33],degrees=True).as_matrix(); s=2.7; shift=np.array([4,-2,8])
        moved=v.copy(); xyz=np.column_stack([v[k] for k in ('x','y','z')])@r.T*s+shift
        for i,k in enumerate(('x','y','z')):moved[k]=xyz[:,i]
        for i in range(3):moved[f'scale_{i}']+=np.log(s)
        q=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
        for i in range(4):moved[f'rot_{i}']=q[i]
        transformed=copy.deepcopy(t);transformed['up']=(r@np.array(t['up'])).tolist()
        normal=r@np.array([0,1,0]); transformed['up_estimation']={'normal':normal.tolist(),'offset':float(-normal@shift)}
        camera=t['reference_camera']; rr=np.asarray(camera['R'])@r.T
        transformed['reference_camera']={'R':rr.tolist(),'t':(s*np.asarray(camera['t'])-rr@shift).tolist()}
        b=build_generation_region(moved,transformed,'A robot arm',DEFAULTS)
        np.testing.assert_allclose(np.asarray(b['bbox_corners']),np.asarray(a['bbox_corners'])@r.T*s+shift,atol=3e-5)
        self.assertEqual(original,v.tobytes())

    def test_mask_retains_seed_protects_rail_and_rejects_uncertain_growth(self):
        cam=Camera(); shape=(100,100); zero=np.zeros(shape,np.float32)
        alpha=zero.copy(); alpha[40:60,40:60]=1
        visible=alpha.copy(); visible[:,49:52]=0
        maps={'target_alpha':alpha,'target_visible_alpha':visible,'target_depth':np.full(shape,5.),
              'target_std':zero.copy(),'scene_depth':np.full(shape,5.),'scene_std':zero.copy(),
              'background_depth':np.full(shape,9.),'background_std':zero.copy(),'background_alpha':np.ones(shape)}
        maps['background_depth'][:,49:52]=3
        maps['scene_depth'][:,49:52]=3
        region={'object_scale':2.,'up':[0,1,0],
            'world_transform':{'center':[0,0,5],'axes':np.eye(3).tolist(),'local_bbox_low':[-1,-1,-1],'local_bbox_high':[1,1,1]},
            'geometry':{'origin':[0,0,5],'axes':np.eye(3).tolist(),'old_local_low':[-.5,-.5,-.5],
                'old_local_high':[.5,.5,.5],'new_local_low':[-1,-1,-1],'new_local_high':[1,1,1],
                'new_side_lengths':[2,2,2]}}
        box=ray_box(cam,region['world_transform'])[0]
        mask,debug,stats=build_mask(cam,maps,region,box,DEFAULTS)
        self.assertTrue(mask[45,45]);self.assertFalse(mask[:,49:52].any())
        self.assertGreater(stats['extension_pixels'],0);self.assertEqual(stats['protected_mask_overlap_pixels'],0)
        uncertain=copy.deepcopy(maps);uncertain['background_std'][:]=2
        smaller,_,report=build_mask(cam,uncertain,region,box,DEFAULTS)
        self.assertEqual(report['extension_pixels'],0)
        self.assertGreater(report['uncertain_extension_rejected_pixels'],0)
        self.assertTrue(smaller[45,45])
        # Broad layers behind the box must not veto useful expansion. A foreground
        # depth prefix still blocks the rail even when its global variance is high.
        uncertain['background_front_mass']=zero.copy();uncertain['background_front_mass'][:,49:52]=.95
        uncertain['background_volume_mass']=zero.copy()
        uncertain['background_old_front_mass']=uncertain['background_front_mass'].copy()
        uncertain['background_surface_depth']=maps['background_depth'].copy()
        corrected,_,report=build_mask(cam,uncertain,region,box,DEFAULTS)
        self.assertGreater(report['extension_pixels'],0)
        self.assertFalse(corrected[:,49:52].any())

    def test_exact_m2_frame_handoff_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'m2').mkdir();(root/'m1').mkdir()
            ctx=SimpleNamespace(stage=lambda n:root/n)
            write_json(root/'m1/manifest.json',{'world_transform':{'legacy':True}})
            self.assertEqual(generation_frame(ctx),{'legacy':True})
            write_json(root/'m2/generation_region.json',{'world_transform':{'adaptive':True}})
            write_json(root/'m2/conditions.json',{'spatial_method':VERSION,
                'generation_region_sha256':sha256(root/'m2/generation_region.json')})
            self.assertEqual(generation_frame(ctx),{'adaptive':True})
            write_json(root/'m2/generation_region.json',{'world_transform':{'corrupted':True}})
            with self.assertRaises(ValueError):generation_frame(ctx)


if __name__=='__main__':unittest.main()
