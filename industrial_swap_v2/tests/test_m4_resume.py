"""CPU checks for exact Adam/camera/RNG continuation; no scene training."""
import copy
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
from industrial_swap_v2.core.local_background_optimizer import optimizer_config
from industrial_swap_v2.core.m4_resume import equal_state,load_resume,restore_training_state,training_signature


class ResumeTests(unittest.TestCase):
    def test_adam_and_camera_sampling_match_uninterrupted_updates(self):
        torch.manual_seed(42)
        model=torch.nn.Linear(2,1); model.register_buffer('permission',torch.tensor([1,0]))
        optimizer=torch.optim.Adam(model.parameters(),lr=.004); rng=np.random.default_rng(42); order=[]
        def update(model,optimizer,rng,order):
            if not order: order.extend(rng.permutation(5).tolist())
            camera=order.pop(); optimizer.zero_grad()
            loss=(model(torch.rand(1,2))-(camera+float(rng.random()))).square().sum()
            loss.backward(); optimizer.step()
            return camera
        for _ in range(8): update(model,optimizer,rng,order)
        saved=copy.deepcopy({'state':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':rng.bit_generator.state,'remaining_view_order':order,
            'torch_rng':torch.get_rng_state(),'cuda_rng':[]})
        expected_order=[update(model,optimizer,rng,order) for _ in range(15)]
        expected=copy.deepcopy(model.state_dict()); expected_optimizer=copy.deepcopy(optimizer.state_dict())
        restored=torch.nn.Linear(2,1); restored.register_buffer('permission',torch.tensor([1,0]))
        other=torch.optim.Adam(restored.parameters(),lr=.9); other_rng=np.random.default_rng(1)
        with patch('torch.cuda.device_count',return_value=0),patch('torch.cuda.set_rng_state_all'):
            other_order=restore_training_state(saved,restored,other,other_rng,5)
        actual_order=[update(restored,other,other_rng,other_order) for _ in range(15)]
        self.assertEqual(expected_order,actual_order)
        self.assertTrue(equal_state(expected,restored.state_dict()))
        self.assertTrue(equal_state(expected_optimizer,other.state_dict()))
        restored.permission.zero_()
        with self.assertRaisesRegex(ValueError,'frozen buffer'):
            restore_training_state(saved,restored,other,other_rng,5)

    def test_resume_checks_lineage_settings_and_full_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'config.yaml'; checkpoint=root/'checkpoint.pth'
            old=optimizer_config({'iterations':2000}); cfg=optimizer_config({'iterations':10000})
            source.write_text(yaml.safe_dump({'project':{'seed':42},'m4':{'local_background':old}}))
            saved={'state':{},'optimizer':{},'step':2000,'signature':training_signature(old,'same_masks'),
                'rng':{},'remaining_view_order':[],'torch_rng':torch.get_rng_state(),'cuda_rng':[],
                'best_state':{},'best_step':0}
            torch.save(saved,checkpoint)
            spec={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                'source_config':str(source),'source_config_sha256':sha256(source)}
            (root/'snapshot.json').write_text(json.dumps({'step':2000,'object_sha256':'device',
                'source_background_sha256':'background','files_sha256':{'checkpoint.pth':sha256(checkpoint)}}))
            ctx=SimpleNamespace(cfg={'project':{'seed':42},'m4':{'resume':spec}},path=Path)
            manifest={'signature':'same_masks','source_hashes':{'object':'device','background':'background'}}
            self.assertEqual(load_resume(ctx,cfg,manifest)['step'],2000)
            for change in ({'color_lr':.01},{'iterations':2000}):
                with self.assertRaises(ValueError): load_resume(ctx,dict(cfg,**change),manifest)
            with self.assertRaises(ValueError): load_resume(ctx,cfg,dict(manifest,signature='changed_masks'))
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'): load_resume(ctx,cfg,manifest)


if __name__=='__main__': unittest.main()
