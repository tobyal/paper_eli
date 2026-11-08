"""Offline regression checks for pre-Qwen case preparation and human handoff."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from industrial_swap_v2.modules import m2_image_generation as m2
from industrial_swap_v2.webapp.adapter import image_import


class CasePreparationTests(unittest.TestCase):
    def context(self, directory):
        folder=Path(directory)/'m2';folder.mkdir()
        cfg={'m1':{'target_text':'a metal storage rack'},'m2':{
            'prompt_mode':'generic_device','replacement_text':'one blue belt conveyor',
            'model':'qwen-image-2.1-pro'}}
        camera={'width':32,'height':32,'fx':20.,'fy':20.,'cx':16.,'cy':16.,'name':'fixture'}
        (folder/'conditions.json').write_text(json.dumps({'camera':camera}))
        return SimpleNamespace(cfg=cfg,stage=lambda name:folder,
            receipt=lambda *args,**kwargs:None,complete=lambda *args,**kwargs:None)

    def test_prepare_only_saves_exact_plan_without_network_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx=self.context(directory)
            with patch.object(m2,'generate',side_effect=AssertionError('API forbidden')) as api, \
                 patch.object(m2,'prepare_conditions',side_effect=AssertionError('Cached conditions')), \
                 patch.object(m2.subprocess,'run',side_effect=AssertionError('No SAM/models')), \
                 patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
                m2.run(ctx,prepare_only=True)
                api.assert_not_called()
            plan=json.loads((ctx.stage('m2')/'generation_plan.json').read_text())
            self.assertEqual(len(plan['images']),3)
            self.assertIn('a metal storage rack',plan['prompt'])
            self.assertIn('one blue belt conveyor',plan['prompt'])
            self.assertIn('UNKNOWN depth',plan['prompt'])
            self.assertNotIn('shoulder, elbow',plan['prompt'])
            self.assertFalse((ctx.stage('m2')/'edited_scene.png').exists())

    def test_existing_robot_prompt_unchanged_without_opt_in(self):
        ctx=SimpleNamespace(cfg={'m2':{'replacement_text':'TEST ROBOT'}})
        prompt=m2.build_prompt(ctx)
        self.assertIn('black assembly and a white support base',prompt)
        self.assertIn('shoulder, elbow, wrist',prompt)
        self.assertIn('TEST ROBOT',prompt)

    def test_prompt_override_is_exact_in_plan_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx=self.context(directory);ctx.cfg['m2']['prompt_override']='Preserve the camera.\nOne blue control cabinet.'
            with patch.object(m2,'generate',side_effect=AssertionError('No API')),patch.object(m2.subprocess,'run',side_effect=AssertionError('No SAM')):
                m2.run(ctx,prepare_only=True)
            self.assertEqual((ctx.stage('m2')/'generation_prompt.txt').read_text(),ctx.cfg['m2']['prompt_override'])
            self.assertEqual(json.loads((ctx.stage('m2')/'generation_plan.json').read_text())['prompt'],ctx.cfg['m2']['prompt_override'])

    def test_manual_rgba_handoff_uses_existing_adapter_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx=self.context(directory);source=Path(directory)/'uploaded.png'
            Image.new('RGBA',(64,64),(100,20,70,125)).save(source)
            with patch('industrial_swap_v2.webapp.adapter.child',side_effect=AssertionError('No subprocess')), \
                 patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
                image_import(ctx,{'asset':{'path':str(source),'kind':'object_rgba','id':'fixture','sha256':'fixture'}})
            self.assertEqual(Image.open(ctx.stage('m2')/'object_mask.png').getextrema(),(125,125))
            camera=json.loads((ctx.stage('m2')/'anchor_camera.json').read_text())
            self.assertEqual((camera['width'],camera['fx']),(64,40.))
            self.assertFalse(json.loads((ctx.stage('m2')/'generation.json').read_text())['external_api_called'])

    def test_manual_image_wrong_aspect_fails_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx=self.context(directory);source=Path(directory)/'wrong.png'
            Image.new('RGB',(64,32)).save(source)
            with patch('industrial_swap_v2.webapp.adapter.child',side_effect=AssertionError('No subprocess')):
                with self.assertRaises(ValueError):
                    image_import(ctx,{'asset':{'path':str(source),'kind':'edited_scene'}})


if __name__=='__main__':unittest.main()
