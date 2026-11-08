"""CPU verification of saved iteration states, without any training or GPU render."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from industrial_swap_v2.core.context import sha256
from industrial_swap_v2.core.scene import read_vertices,save_vertices
from industrial_swap_v2.core.m4_snapshots import export_background,save_iteration_result


class SnapshotTests(unittest.TestCase):
    def fixture(self):
        names=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        rows=np.zeros(4,dtype=[(n,'f4') for n in names]); rows['x']=np.arange(4); rows['rot_0']=1
        model=SimpleNamespace(indices=torch.tensor([0,1]),rgb_indices=torch.tensor([0]),
            initial_logits=torch.zeros(2,1),opacity_delta=torch.tensor([[-.3],[-.4]]),
            dc_delta=torch.tensor([[.1,.2,.3]]),original=object(),state_dict=lambda:{'state':torch.tensor([1.])})
        return rows,model

    def test_export_preserves_geometry_and_frozen_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); rows,model=self.fixture(); source=root/'source.ply'; save_vertices(source,rows)
            digest=sha256(source); checks=export_background(model,source,root/'result.ply')
            result=read_vertices(root/'result.ply')
            self.assertEqual(result[2:].tobytes(),rows[2:].tobytes())
            self.assertEqual(result['x'].tobytes(),rows['x'].tobytes())
            self.assertEqual(result['f_dc_0'][1:].tobytes(),rows['f_dc_0'][1:].tobytes())
            np.testing.assert_allclose(result['opacity'][:2],[-.3,-.4])
            self.assertEqual(sha256(source),digest); self.assertEqual(checks['deleted_points'],0)

    def test_snapshot_is_actual_state_immutable_and_has_checkpoint_metrics_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for stage in ('m1','m3','m4'): (root/stage).mkdir()
            rows,model=self.fixture(); source=root/'m1/background_gaussians.ply'; objpath=root/'m3/object_world.ply'
            save_vertices(source,rows); save_vertices(objpath,rows[:1]); original_sha=sha256(source); obj_sha=sha256(objpath)
            ctx=SimpleNamespace(stage=lambda name:root/name)
            common=SimpleNamespace(GaussianData=SimpleNamespace(load=lambda path:read_vertices(path)),
                render_joint=lambda *a,**k:{'rgb':torch.zeros(3,8,8)},
                save_image=lambda path,data:Image.new('RGB',(8,8)).save(path))
            views=[{'split':split,'name':split,'camera':{'name':split},'camera_object':None} for split in ('anchor','validation')]
            optimizer=SimpleNamespace(state_dict=lambda:{'step':500}); rng=np.random.default_rng(42)
            args=(ctx,model,None,views,common,500,{'acceptable':False},optimizer,rng,[2,1],'signature',{},0)
            with patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.empty_cache'):
                out=save_iteration_result(*args)
                with self.assertRaises(FileExistsError): save_iteration_result(*args)
            saved=json.loads((out/'snapshot.json').read_text()); self.assertFalse(saved['guard_passed'])
            for name,digest in saved['files_sha256'].items(): self.assertEqual(sha256(out/name),digest)
            self.assertTrue((out/'checkpoint.pth').is_file()); self.assertTrue((out/'anchor_comparison.png').is_file())
            scene=read_vertices(out/'scene.ply'); bg=read_vertices(out/'background_optimized.ply')
            self.assertEqual(scene[:4].tobytes(),bg.tobytes()); self.assertEqual(scene[4:].tobytes(),rows[:1].tobytes())
            self.assertEqual(sha256(source),original_sha); self.assertEqual(sha256(objpath),obj_sha)


if __name__=='__main__': unittest.main()
