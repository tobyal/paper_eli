"""CPU-only compatibility/regression checks; no external models or GPU jobs."""
import ast
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from industrial_swap_v2.core.context import sha256
from industrial_swap_v2.core.workflow import current_config,validate_workflow,METHOD
from industrial_swap_v2.core.multiview_masks import mask_config,cache_matches,local_support,ensure_m4_supervision
from industrial_swap_v2.core.local_background_optimizer import optimizer_config
from industrial_swap_v2.core.visibility_loss import masked_l1,visibility_loss
from industrial_swap_v2.core.m4_outputs import compose_final_scene
from industrial_swap_v2.core.scene import save_vertices,read_vertices
from industrial_swap_v2.tests.gaussian_fixture import Common

BASE=Path(__file__).parents[1]
ARCHIVE=BASE/'legacy/pre_four_module_cleanup'


def tree(path): return ast.parse(path.read_text(encoding='utf-8'))
def node(path,name): return next(n for n in tree(path).body if getattr(n,'name',None)==name)


class WorkflowTests(unittest.TestCase):
    def test_all_official_modules_import_without_diffusion(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if any(n in name.lower() for n in ('diffusion','diffusers','controlnet','zero123')):
                raise AssertionError('Unexpected diffusion import: '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            for module in ('m1_localization','m2_image_generation','m3_trellis_generation','m4_refinement'):
                importlib.reload(importlib.import_module('industrial_swap_v2.modules.'+module))

    def test_default_config_is_only_official_flow(self):
        cfg=yaml.safe_load((BASE/'configs/default.yaml').read_text())
        validate_workflow(cfg)
        self.assertEqual(cfg['m4']['local_background']['iterations'],250)
        self.assertEqual(set(cfg['m4']),{'method','integration','minimum_free_gpu_gib','multiview_supervision','local_background','save_every'})
        self.assertEqual(cfg['m3']['method'],'pbr_multiview_gaussian')
        self.assertNotIn('object_occlusion_filter',cfg['m3'])
        self.assertNotIn('visibility_initialization',cfg['m3'])
        self.assertNotEqual(cfg['project']['output_dir'],'outputs')

    def test_migration_does_not_mutate_source_or_generation_settings(self):
        old=yaml.safe_load((ARCHIVE/'configs/default.yaml').read_text()); saved=copy.deepcopy(old)
        new=current_config(old); self.assertEqual(old,saved)
        self.assertEqual(new['m1'],old['m1'])
        self.assertEqual({k:v for k,v in new['m2'].items() if k!='spatial_conditions'},
                         {k:v for k,v in old['m2'].items() if k!='spatial_conditions'})
        self.assertEqual(new['m2']['spatial_conditions']['method'],'adaptive_visible_v1')
        for key in old['m3']:
            if key not in ('object_occlusion_filter','visibility_initialization','output_type','use_original_scene_transform'):
                self.assertEqual(new['m3'][key],old['m3'][key])
        self.assertEqual(new['m4']['method'],METHOD)
        self.assertEqual(new['m4']['multiview_supervision']['scheme'],'old_new_union_v2')
        self.assertNotIn('visibility_initialization',new['m3'])
        with self.assertRaises(ValueError): validate_workflow(old)

    def test_m3_composition_has_no_mask_or_m4_configuration_dependency(self):
        from industrial_swap_v2.core.m3_composition import compose_m3
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'m1').mkdir(); (root/'m3').mkdir()
            vertices=np.zeros(3,dtype=[('x','f4'),('y','f4'),('z','f4')]); vertices['x']=np.arange(3)
            save_vertices(root/'m1/background_gaussians.ply',vertices[:2])
            save_vertices(root/'m3/object_world.ply',vertices[2:])
            ctx=SimpleNamespace(cfg={'m3':{}},stage=lambda name:root/name)
            with patch('industrial_swap_v2.core.multiview_masks.prepare_multiview',side_effect=AssertionError('M3 must not build supervision')) as prepare:
                compose_m3(ctx); prepare.assert_not_called()
            self.assertEqual(read_vertices(root/'m3/scene_coarse.ply').tobytes(),vertices.tobytes())
            self.assertFalse((root/'m4').exists()); self.assertFalse((root/'m3/multiview_visibility').exists())

    def test_new_supervision_is_written_to_m4_and_old_m3_cache_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            paths={'background':'m1/background_gaussians.ply','object':'m3/object_world.ply',
                'manifest':'m1/manifest.json','mask':'m2/object_mask.png','camera':'m2/anchor_camera.json',
                'original_scene':'source.ply','target_indices':'m1/target_indices.npy'}
            for value in paths.values():
                p=root/value; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'fixture')
            cfg=yaml.safe_load((BASE/'configs/default.yaml').read_text()); cfg['project']['source_ply']='source.ply'
            cfg['m3'].pop('method',None)  # This fixture verifies historical cache compatibility.
            ctx=SimpleNamespace(cfg=cfg,stage=lambda name:root/name,source=lambda name:root/name)
            destination=root/'m4/multiview_supervision'
            with patch('industrial_swap_v2.core.multiview_masks.prepare_multiview',return_value=destination) as prepare:
                self.assertEqual(ensure_m4_supervision(ctx),destination)
                prepare.assert_called_once_with(ctx,destination)
            old=root/'m3/multiview_visibility'; old.mkdir()
            manifest={'permission_scheme':'old_new_union_v2','config':mask_config(cfg['m4']['multiview_supervision']),
                'source_hashes':{key:sha256(root/value) for key,value in paths.items()}}
            saved=old/'manifest.json'; saved.write_text(json.dumps(manifest)); before=saved.read_bytes()
            with patch('industrial_swap_v2.core.multiview_masks.prepare_multiview',side_effect=AssertionError('Matching old cache must be read-only')):
                self.assertEqual(ensure_m4_supervision(ctx),old)
            self.assertEqual(saved.read_bytes(),before)

    def test_effective_numerical_defaults_match_previous_union(self):
        path=ARCHIVE/'core/local_background_optimizer.py'
        old_default=next(n.value for n in tree(path).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULTS' for t in n.targets))
        cfg_node=next(n.value for n in node(path,'optimize_permissioned_background').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in n.targets))
        previous=eval(compile(ast.Expression(cfg_node),str(path),'eval'),{'DEFAULTS':ast.literal_eval(old_default),'ctx':SimpleNamespace(cfg={'m4':{}})})
        for k in ('outside_preserve_weight','candidate_max_opacity_change'): previous.pop(k)
        self.assertEqual(previous,optimizer_config())

    def test_mask_permission_and_loss_math_unchanged(self):
        functions={'core/multiview_masks.py':['union_supervision_masks','occlusion_permissions','ray_box_depth','local_support','choose_views'],
            'core/local_background_optimizer.py':['PermissionedBackground','evaluate_permissioned'],
            'core/visibility_loss.py':['masked_l1','visibility_loss','object_contribution']}
        for file,names in functions.items():
            for name in names:
                self.assertEqual(ast.dump(node(BASE/file,name)),ast.dump(node(ARCHIVE/file,name)),(file,name))

    def test_entire_training_loop_unchanged(self):
        before=node(ARCHIVE/'core/local_background_optimizer.py','optimize_permissioned_background')
        after=node(BASE/'core/local_background_optimizer.py','optimize_background')
        loop=lambda f:next(n for n in f.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
        updated=copy.deepcopy(loop(after))
        # Resume changes only the absolute starting iteration; updates stay exact.
        updated.iter=copy.deepcopy(loop(before).iter)
        # Iteration snapshots are output-only; the numerical update loop stays exact.
        updated.body=[n for n in updated.body if not (isinstance(n,ast.If) and
            any(isinstance(x,ast.Name) and x.id=='save_every' for x in ast.walk(n.test)))]
        self.assertEqual(ast.dump(loop(before)),ast.dump(updated))

    def test_union_cache_compatibility_rejects_legacy_or_changed_threshold(self):
        cfg=mask_config(); hashes={'object':'example'}
        old=dict(cfg,edge_dilation_fraction=.008)
        saved={'permission_scheme':'old_new_union_v2','config':old,'source_hashes':hashes}
        self.assertTrue(cache_matches(saved,cfg,hashes))
        self.assertFalse(cache_matches(saved,cfg,{'object':'changed'}))
        self.assertFalse(cache_matches(saved,dict(cfg,new_alpha_threshold=.3),hashes))
        saved['permission_scheme']='legacy_bbox'; self.assertFalse(cache_matches(saved,cfg,hashes))

    def test_final_scene_uses_optimized_background_and_exact_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'m3').mkdir(); (root/'m4/background_optimization').mkdir(parents=True)
            names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
            vertices=np.zeros(5,dtype=[(n,'f4') for n in names]); vertices['x']=np.arange(5); vertices['rot_0']=1
            bg=root/'m4/background_optimization/background_optimized.ply'; obj=root/'m3/object_world.ply'
            save_vertices(bg,vertices[:2]); save_vertices(obj,vertices[2:])
            summary={'background_sha256':sha256(bg),'object_sha256':sha256(obj)}
            (bg.parent/'summary.json').write_text(json.dumps(summary))
            ctx=SimpleNamespace(stage=lambda name:root/name)
            compose_final_scene(ctx,summary)
            self.assertEqual(read_vertices(root/'m4/scene_final.ply').tobytes(),vertices.tobytes())
            self.assertEqual(sha256(obj),sha256(root/'m4/object_fixed.ply'))
            bg.write_bytes(b'changed')
            with self.assertRaises(ValueError): compose_final_scene(ctx,summary)

    def test_visibility_gradients_protect_legal_foreground_and_fixed_device(self):
        def data(alpha):
            return Common.GaussianData(torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,4),torch.logit(torch.tensor([[alpha]])),torch.zeros(1,3,1))
        bg=data(.5); obj=data(.8); bg.opacity_logit.requires_grad_(True)
        loss,visible=visibility_loss(Common,None,bg,obj,torch.tensor([[[.8]]]),torch.ones(1,1))
        self.assertAlmostEqual(float(visible),.4,places=6); loss.backward()
        self.assertGreater(float(bg.opacity_logit.grad),0); self.assertIsNone(obj.opacity_logit.grad)
        bg.opacity_logit.grad.zero_()
        loss,_=visibility_loss(Common,None,bg,obj,torch.tensor([[[.8]]]),torch.zeros(1,1)); loss.backward()
        self.assertEqual(float(bg.opacity_logit.grad),0)

    def test_environment_rgb_does_not_learn_excluded_pixels(self):
        rgb=torch.zeros(3,2,2,requires_grad=True); mask=torch.zeros(2,2); mask[0,0]=1
        masked_l1(rgb,torch.ones_like(rgb),mask).backward()
        self.assertTrue(torch.equal(rgb.grad[:,1],torch.zeros(3,2)))

    def test_covariance_support_includes_centers_outside_region(self):
        names=['x','y','z']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        v=np.zeros(3,dtype=[(n,'f4') for n in names]); v['x']=[0,1,4]; v['rot_0']=1
        for name in ('scale_0','scale_1','scale_2'): v[name]=np.log(.01)
        v['scale_0'][1]=np.log(.3)
        frame={'center':[0,0,0],'axes':np.eye(3).tolist(),'local_bbox_low':[-.5]*3,'local_bbox_high':[.5]*3}
        ids,_=local_support(v,{'world_transform':frame},0); self.assertEqual(ids.tolist(),[0,1])


if __name__=='__main__': unittest.main()
