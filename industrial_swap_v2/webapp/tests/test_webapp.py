"""CPU-only regression tests against an isolated database and tiny fixtures."""
import io
import json
import os
import struct
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
os.environ['SWAP_WEB_DATA']=str(ROOT/'industrial_swap_v2/webapp/data'/('test_'+uuid.uuid4().hex))
from fastapi.testclient import TestClient
from industrial_swap_v2.webapp import store as s, service as v
from industrial_swap_v2.webapp.app import app

CAM={'R':np.eye(3).tolist(),'t':[0,0,1],'width':64,'height':64,'fx':60,'fy':60,'cx':32,'cy':32,'name':'test'}


def png(rgba=True,size=(64,64)):
    im=Image.new('RGBA' if rgba else 'RGB',size,(255,255,255,0) if rgba else 'white')
    if rgba:
        for x in range(20,40):
            for y in range(15,50): im.putpixel((x,y),(250,210,20,255))
    out=io.BytesIO(); im.save(out,format='PNG'); return out.getvalue()


def fixture(exp,stage='m2',action='prepare'):
    rid=s.uid(); folder=s.DATA/'experiments'/exp/'runs'/rid; directory=folder/'outputs'/stage; directory.mkdir(parents=True)
    (directory/'anchor_camera.json').write_text(json.dumps(CAM)); (directory/'conditions.json').write_text(json.dumps({'camera':CAM}))
    (directory/'anchor.png').write_bytes(png(False))
    outputs={'directory':str(directory),'files':['anchor_camera.json','conditions.json','anchor.png'],
        'integrity':{n:{'sha256':s.sha(directory/n),'bytes':(directory/n).stat().st_size} for n in ('anchor_camera.json','conditions.json','anchor.png')}}
    with s.transaction() as conn:
        s.sql(conn,'''INSERT INTO stage_runs(id,experiment_id,stage,action,status,signature,inputs,parameters,config_path,folder,outputs,log_path,created_at,request_id)
            VALUES(:id,:e,:stage,:action,'completed',:sig,'{}','{}',:cfg,:folder,:out,:log,:t,:id)''',
            id=rid,e=exp,stage=stage,action=action,sig=s.digest(outputs),cfg=str(folder/'config.yaml'),folder=str(folder),out=s.dump(outputs),log=str(folder/'log'),t=s.now())
    return rid


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app,headers={'X-Swap-Action':'1'})
        self.exp=v.create('test '+s.uid()[:8]); self.material=fixture(self.exp)
        self.m1=fixture(self.exp,'m1','reuse')
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET heads=:h WHERE id=:e',h=s.dump({'m1':self.m1,'materials':self.material}),e=self.exp)

    def upload(self,data=None,**params):
        return self.client.post('/api/assets/upload',data={'experiment_id':self.exp,'camera_aligned':'true',
            'kind':'object_rgba',**params},files={'file':('fixture.png',data or png(),'image/png')})

    def test_no_cross_origin_or_unmarked_mutation(self):
        self.assertEqual(TestClient(app).post('/api/experiments',json={'name':'unsafe'}).status_code,403)
        self.assertEqual(self.client.post('/api/experiments',json={'name':'unsafe'},headers={'Origin':'https://other.example'}).status_code,403)

    def test_upload_deduplicates_files_and_keeps_usage(self):
        first=self.upload().json(); second=self.upload().json()
        self.assertEqual(first['id'],second['id']); self.assertTrue(second['deduplicated'])
        self.assertGreaterEqual(len(second['events']),2)
        self.assertEqual(s.asset_path(first['id']).read_bytes(),png())

    def test_rgba_requires_alpha_and_matching_canvas(self):
        self.assertEqual(self.upload(png(False)).status_code,422)
        self.assertEqual(self.upload(png(size=(80,64))).status_code,422)
        self.assertEqual(self.upload(camera_aligned='false').status_code,422)

    def test_selection_preserves_history_and_invalidates_downstream(self):
        old=fixture(self.exp,'m3','reuse'); aid=self.upload().json()['id']
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET heads=:h WHERE id=:e',h=s.dump({'m1':self.m1,'materials':self.material,'m3':old,'m4':old}),e=self.exp)
        response=self.client.post('/api/experiments/'+self.exp+'/selection',json={'asset_id':aid,'role':'m2'})
        self.assertEqual(response.status_code,200); self.assertNotIn('m3',response.json()['heads'])
        self.assertTrue(v.stage_dir(old).exists()); self.assertEqual(v.run(old)['status'],'completed')

    def test_incompatible_camera_is_rejected(self):
        aid=self.upload().json()['id']
        metadata=s.unpack(s.asset(aid)['metadata']); metadata['camera_key']='different'
        # Use a distinct record so other fixtures remain valid.
        with s.transaction() as conn: s.sql(conn,'UPDATE assets SET metadata=:m WHERE id=:id',m=s.dump(metadata),id=aid)
        v.select(self.exp,aid,'m2')
        response=self.client.post('/api/experiments/'+self.exp+'/runs',json={'stage':'m2','action':'reuse','request_id':s.uid()})
        self.assertEqual(response.status_code,422)
        metadata['camera_key']=v.camera_key(CAM)
        with s.transaction() as conn: s.sql(conn,'UPDATE assets SET metadata=:m WHERE id=:id',m=s.dump(metadata),id=aid)

    def test_qwen_requires_explicit_confirmation_and_is_idempotent(self):
        body={'stage':'m2','action':'qwen','request_id':s.uid()}
        self.assertEqual(self.client.post('/api/experiments/'+self.exp+'/runs',json=body).status_code,422)
        body['confirm_qwen']=True
        first=self.client.post('/api/experiments/'+self.exp+'/runs',json=body).json()
        second=self.client.post('/api/experiments/'+self.exp+'/runs',json=body).json()
        self.assertEqual(first['id'],second['id'])
        self.assertFalse(list(s.DATA.rglob('qwen_generation_once.lock')))
        body['parameters']={'force_model':True}
        self.assertEqual(self.client.post('/api/experiments/'+self.exp+'/runs',json=body).status_code,422)

    def test_missing_m4_reference_is_rejected(self):
        response=self.client.post('/api/experiments/'+self.exp+'/runs',json={'stage':'m4','action':'refine','request_id':s.uid()})
        self.assertEqual(response.status_code,422)

    def test_file_endpoint_rejects_path_traversal(self):
        response=self.client.get('/api/runs/'+self.material+'/file',params={'name':'../../config.yaml'})
        self.assertEqual(response.status_code,404)

    def test_glb_cannot_reference_external_files(self):
        scene=json.dumps({'asset':{'version':'2.0'},'buffers':[{'uri':'file:///etc/passwd','byteLength':1}]}).encode()
        scene+=b' '*((-len(scene))%4)
        glb=struct.pack('<4sII',b'glTF',2,20+len(scene))+struct.pack('<II',len(scene),0x4e4f534a)+scene
        response=self.client.post('/api/assets/upload',data={'kind':'mesh'},files={'file':('bad.glb',glb,'model/gltf-binary')})
        self.assertEqual(response.status_code,422)
        self.assertIn('自包含',response.json()['detail'])

    def test_world_ply_needs_source_transform(self):
        for transform in ('','[]','{"rotation":"bad"}','{"scale":{},"rotation":null}'):
            with self.subTest(transform=transform):
                response=self.client.post('/api/assets/upload',data={'kind':'gaussian_world','experiment_id':self.exp,'transform_json':transform},
                    files={'file':('world.ply',b'ply\n','application/octet-stream')})
                self.assertEqual(response.status_code,422)
                self.assertIn('变换',response.json()['detail'])

    def test_asset_preserves_full_prepared_prompt(self):
        import yaml
        value=v.run(self.material); directory=v.stage_dir(self.material)
        Path(value['config_path']).write_text(yaml.safe_dump(s.config()))
        full_prompt='Keep the original camera and surroundings. Generate a yellow robotic arm.'
        (directory/'generation_plan.json').write_text(json.dumps({'prompt':full_prompt}))
        assets=v.register_outputs(value,directory)
        self.assertEqual(s.asset(assets['anchor.png'])['prompt'],full_prompt)

    def test_cache_integrity_checks_content_not_just_size(self):
        value=v.run(self.material); self.assertTrue(v.verify_outputs(value))
        path=v.stage_dir(self.material)/'anchor.png'; data=bytearray(path.read_bytes()); data[-1]^=1; path.write_bytes(data)
        self.assertFalse(v.verify_outputs(value))

    def test_cache_changes_when_identical_parameters_produce_different_pixels(self):
        first=v.input_fingerprint('m2',self.material)
        row=v.run(self.material); outputs=s.unpack(row['outputs'])
        outputs['integrity']['anchor.png']['sha256']='a-different-generated-image'
        with s.transaction() as conn:
            s.sql(conn,'UPDATE stage_runs SET outputs=:out WHERE id=:id',out=s.dump(outputs),id=self.material)
        second=v.input_fingerprint('m2',self.material)
        self.assertEqual(first['signature'],second['signature'])
        self.assertNotEqual(s.digest(first),s.digest(second))

    def test_resume_partial_directory_does_not_claim_stage_cache(self):
        from industrial_swap_v2.webapp.worker import execute
        rid=v.submit(self.exp,'m2','prepare',s.uid())
        value=v.run(rid); (Path(value['folder'])/'outputs/m2').mkdir(parents=True)
        with patch('industrial_swap_v2.webapp.worker.s.rows') as cache_candidates, \
             patch('industrial_swap_v2.webapp.worker.choose_gpu',return_value=None):
            self.assertFalse(execute(value))
            cache_candidates.assert_not_called()

    def test_asset_reuse_does_not_reserve_inference_memory(self):
        from industrial_swap_v2.webapp.worker import choose_gpu
        cfg=s.config()
        cfg['m3']['object_occlusion_filter']={'enabled':False}
        cfg['m3']['visibility_initialization']={'enabled':False}
        with patch('industrial_swap_v2.webapp.worker.subprocess.check_output',return_value='0, 1024\n1, 6144\n'):
            self.assertEqual(choose_gpu(cfg,{'stage':'m1','action':'reuse','inputs':'{}'}),1)
            self.assertEqual(choose_gpu(cfg,{'stage':'m3','action':'reuse','inputs':'{}'}),1)
            with patch('industrial_swap_v2.webapp.worker.generation_cache',return_value=None):
                self.assertIsNone(choose_gpu(cfg,{'stage':'m3','action':'trellis','inputs':'{}'}))

    def test_filtering_reuse_requires_attribution_memory(self):
        from industrial_swap_v2.webapp.worker import choose_gpu
        cfg=s.config(); cfg['m3']['object_occlusion_filter']={'enabled':True}
        with patch('industrial_swap_v2.webapp.worker.subprocess.check_output',return_value='0, 1024\n1, 6144\n'):
            self.assertIsNone(choose_gpu(cfg,{'stage':'m3','action':'reuse','inputs':'{}'}))

    def test_multiview_reuse_reserves_rendering_memory(self):
        from industrial_swap_v2.webapp.worker import choose_gpu
        cfg=s.config(); cfg['m3']['object_occlusion_filter']={'enabled':False}
        cfg['m3']['visibility_initialization']={'enabled':True}
        with patch('industrial_swap_v2.webapp.worker.subprocess.check_output',return_value='0, 1024\n1, 6144\n'):
            self.assertIsNone(choose_gpu(cfg,{'stage':'m3','action':'reuse','inputs':'{}'}))

    def test_filter_options_reject_floor_repair_and_unknown_parameters(self):
        for settings in ({'floor_repair':True},{'yellow_protection':True},{'enabled':'yes'}):
            response=self.client.post('/api/experiments/'+self.exp+'/runs',json={
                'stage':'m3','action':'reuse','request_id':s.uid(),
                'parameters':{'object_occlusion_filter':settings}})
            self.assertEqual(response.status_code,422)

    def test_rgba_adapter_skips_sam_and_preserves_alpha(self):
        from industrial_swap_v2.core.context import Context
        from industrial_swap_v2.webapp.adapter import image_import
        import yaml
        folder=s.DATA/'experiments'/self.exp/'rgba_adapter'; (folder/'outputs/m2').mkdir(parents=True)
        cfg=s.config(); cfg['project'].update(output_dir=str((folder/'outputs').relative_to(s.BASE)),
            records_dir=str((folder/'records').relative_to(s.BASE)),logs_dir=str((folder/'logs').relative_to(s.BASE)))
        cp=folder/'config.yaml'; cp.write_text(yaml.safe_dump(cfg)); ctx=Context(cp)
        (ctx.stage('m2')/'conditions.json').write_text(json.dumps({'camera':CAM}))
        source=folder/'rgba.png'; source.write_bytes(png())
        with patch('industrial_swap_v2.webapp.adapter.child') as child:
            image_import(ctx,{'asset':{'path':str(source),'kind':'object_rgba','id':'test','sha256':s.sha(source)}})
            child.assert_not_called()
        original=np.asarray(Image.open(source).getchannel('A'))
        self.assertTrue(np.array_equal(original,np.asarray(Image.open(ctx.stage('m2')/'object_mask.png'))))
        self.assertFalse((ctx.records/'qwen_generation_once.lock').exists())


if __name__=='__main__': unittest.main(verbosity=2)
