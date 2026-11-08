"""CPU-only case handoff tests; isolated database, no AVP/model execution."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import yaml
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[3]
os.environ['SWAP_WEB_DATA']=str(ROOT/'industrial_swap_v2/webapp/data'/('test_m2_'+uuid.uuid4().hex))
from industrial_swap_v2.webapp import store as s,service as v,m1_cases,m2_cases as m
from industrial_swap_v2.webapp.app import app
from industrial_swap_v2.core.context import write_json,read_json


class M2CaseTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app,headers={'X-Swap-Action':'1'});self.cid=s.uid();jid=s.uid()
        self.folder=m1_cases.HOME/'validation_cases'/self.cid;out=self.folder/'outputs/m1';out.mkdir(parents=True)
        cfg=s.config();cfg['m1']['target_text']='a test industrial cabinet'
        (self.folder/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
        write_json(self.folder/'confirmed_case.json',{'id':self.cid})
        write_json(self.folder/'localization/target.json',{'text':cfg['m1']['target_text']})
        write_json(out/'manifest.json',{'counts':{'source':3,'target':1,'background':2},'world_transform':{'fixture':True}})
        for name in ('edit_region.json','target.json'):write_json(out/name,{'fixture':self.cid})
        for name in ('target_gaussians.ply','background_gaussians.ply','target_indices.npy','keep_indices.npy'):(out/name).write_bytes((self.cid+name).encode())
        write_json(self.folder/'integrity.json',{str(p.relative_to(self.folder)):s.sha(p) for p in self.folder.rglob('*') if p.is_file()})
        with s.transaction() as conn:
            s.sql(conn,'''INSERT INTO m1_jobs(id,selection,text,status,folder,validation,request_id,created_at)
                VALUES(:id,'fixture','a test cabinet','completed',:folder,1,:id,:t)''',id=jid,folder=str(self.folder),t=s.now())
            s.sql(conn,'INSERT INTO m1_cases VALUES(:id,:job,:name,:folder,1,:t)',id=self.cid,job=jid,name='fixture case',folder=str(self.folder),t=s.now())
        self.url=f'/api/m1-assist/cases/{self.cid}/m2'

    def test_prompt_uses_this_case_and_new_description_without_queue(self):
        before=len(s.rows('SELECT id FROM stage_runs'))
        response=self.client.post(self.url+'/prompt',json={'replacement_text':'A blue tool cabinet.'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('a test industrial cabinet',response.json()['prompt']);self.assertIn('A blue tool cabinet.',response.json()['prompt'])
        self.assertEqual(len(s.rows('SELECT id FROM stage_runs')),before)

    def test_description_options_are_case_bound_and_never_queue(self):
        before=len(s.rows('SELECT id FROM stage_runs'))
        before_files={str(p):s.sha(p) for p in self.folder.rglob('*') if p.is_file()}
        options=[{'id':letter,'label':'fixture '+letter,'description':'A compact '+color+' industrial cabinet.'}
                 for letter,color in zip('ABC',('blue','red','green'))]
        catalog={'cases':[{'case_id':self.cid,'options':options,'design_note':'Fixture'},
                          {'case_id':s.uid(),'options':[{'id':'OTHER'}]}]}
        read=m.read_json
        def fixture(path):
            return catalog if Path(path).name=='m2_case_description_options.json' else read(path)
        with patch.object(m,'read_json',side_effect=fixture):
            response=self.client.get(self.url+'/description-options')
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['options'],options)
            for option in options:
                prompt=self.client.post(self.url+'/prompt',json={'replacement_text':option['description']})
                self.assertEqual(prompt.status_code,200,prompt.text)
                self.assertIn(option['description'],prompt.json()['prompt'])
            missing=self.client.get(f'/api/m1-assist/cases/{s.uid()}/m2/description-options')
            self.assertEqual(missing.status_code,404)
        self.assertEqual(len(s.rows('SELECT id FROM stage_runs')),before)
        self.assertEqual({str(p):s.sha(p) for p in self.folder.rglob('*') if p.is_file()},before_files)
        # A future/validation case without written proposals stays editable, with no fallback robot.
        self.assertEqual(self.client.get(self.url+'/description-options').json()['options'],[])

    def test_immutable_import_and_prepare_only_handoff(self):
        request={'replacement_text':'A blue tool cabinet.','request_id':s.uid(),'prompt_override':'Exact custom prompt, preserve the camera.'}
        response=self.client.post(self.url+'/prepare',json=request);self.assertEqual(response.status_code,200,response.text)
        prepared=response.json();run=v.run(prepared['run_id']);self.assertEqual((run['stage'],run['action']),('m2','prepare'))
        cfg=yaml.safe_load(Path(run['config_path']).read_text());self.assertFalse(cfg['m2']['reuse_anchor'])
        self.assertTrue(cfg['m2']['preparation_only']);self.assertEqual(cfg['m2']['confirmed_m1_case_id'],self.cid)
        self.assertEqual(cfg['m2']['prompt_override'],request['prompt_override'])
        inputs=s.unpack(run['inputs']);imported=v.run(inputs['m1']);self.assertEqual(imported['action'],'case_import')
        self.assertEqual((v.stage_dir(imported['id'])/'background_gaussians.ply').read_bytes(),(self.folder/'outputs/m1/background_gaussians.ply').read_bytes())
        self.assertEqual(self.client.post(self.url+'/prepare',json=request).json()['id'],prepared['id'])
        request['replacement_text']='A green tool cabinet.'
        self.assertEqual(self.client.post(self.url+'/prepare',json=request).status_code,422)
        with s.transaction() as conn:s.sql(conn,"UPDATE stage_runs SET status='failed' WHERE id=:id",id=run['id'])
        # Give Qwen its normal prerequisites: the preparation-only guard must still reject it.
        with s.transaction() as conn:
            s.sql(conn,'UPDATE experiments SET heads=:heads WHERE id=:id',heads=s.dump({'m1':imported['id'],'materials':imported['id']}),id=prepared['experiment_id'])
        with self.assertRaisesRegex(ValueError,'仅准备 Qwen 输入'):
            v.submit(prepared['experiment_id'],'m2','qwen',s.uid(),confirm_qwen=True)

    def test_tampered_m1_fails_before_creating_preparation(self):
        (self.folder/'outputs/m1/edit_region.json').write_text('{}')
        response=self.client.post(self.url+'/prepare',json={'replacement_text':'A blue machine','request_id':s.uid()})
        self.assertEqual(response.status_code,422);self.assertIn('校验失败',response.text)
        self.assertEqual(self.client.get(self.url).json(),[])

    def test_unconfirmed_case_and_blank_prompt_rejected(self):
        response=self.client.post(f'/api/m1-assist/cases/{s.uid()}/m2/prepare',json={'replacement_text':'A test cabinet','request_id':s.uid()})
        self.assertEqual(response.status_code,404)
        response=self.client.post(self.url+'/prepare',json={'replacement_text':'A test cabinet','request_id':s.uid(),'prompt_override':'   '})
        self.assertEqual(response.status_code,422)

    def test_archived_case_hidden_and_not_submitted(self):
        with s.transaction() as conn:
            s.sql(conn,'INSERT INTO m1_case_archives VALUES(:id,:reason,:t)',id=self.cid,reason='User excluded duplicate',t=s.now())
        self.assertNotIn(self.cid,{r['id'] for r in self.client.get('/api/m1-assist/cases?validation=true').json()})
        self.assertIn(self.cid,{r['id'] for r in self.client.get('/api/m1-assist/cases?validation=true&include_archived=true').json()})
        response=self.client.post(self.url+'/prepare',json={'replacement_text':'A new cabinet','request_id':s.uid()})
        self.assertEqual(response.status_code,422);self.assertTrue((self.folder/'integrity.json').is_file())

    def test_condition_cache_requires_identical_case_geometry_and_settings(self):
        import copy
        from industrial_swap_v2.webapp import m2_condition_cache as cache
        source=self.folder/'cache_fixture';source.mkdir();m1=self.folder/'outputs/m1';out=self.folder/'reuse_fixture'
        cfg=s.config();scene=self.folder/'scene.ply';scene.write_bytes(b'original scene fixture')
        cfg['project']['source_ply']=str(scene);cfg['project']['source_sha256']=s.sha(scene)
        cfg['m2']['confirmed_m1_case_id']=self.cid
        cp=self.folder/'cache_config.yaml';cp.write_text(yaml.safe_dump(cfg))
        for name in cache.CONDITION_FILES:
            if name!='conditions.json':(source/name).write_bytes(('unchanged '+name).encode())
        hashes={n:s.sha(m1/n) for n in cache.M1_FILES}
        write_json(source/'conditions.json',{'m1_case_id':self.cid,'input_sha256':hashes,
            'spatial_method':cache.VERSION,'condition_signature':cache.signature(cfg),
            'source_scene_sha256':s.sha(scene),'source_anchor':str(source/'avp_work/avp/anchor.json')})
        next_cfg=copy.deepcopy(cfg)
        row={'id':'fixture-source','config_path':str(cp)}
        with patch.object(s,'rows',return_value=[row]),patch.object(v,'verify_outputs',return_value=True),patch.object(v,'stage_dir',return_value=source):
            different=copy.deepcopy(cfg);different['m2']['replacement_text']='A different device'
            self.assertIsNone(cache.reuse_conditions(different,out,m1));self.assertFalse(out.exists())
            wrong=copy.deepcopy(next_cfg);wrong['project']['seed']+=1
            self.assertIsNone(cache.reuse_conditions(wrong,out,m1));self.assertFalse(out.exists())
            original=(m1/'edit_region.json').read_bytes();(m1/'edit_region.json').write_bytes(b'different geometry')
            self.assertIsNone(cache.reuse_conditions(next_cfg,out,m1));self.assertFalse(out.exists())
            (m1/'edit_region.json').write_bytes(original)
            self.assertEqual(cache.reuse_conditions(next_cfg,out,m1),'fixture-source')
        self.assertEqual((out/'anchor.png').read_bytes(),(source/'anchor.png').read_bytes())
        self.assertFalse((out/'generation_prompt.txt').exists())
        self.assertEqual(read_json(out/'conditions.json')['condition_cache']['source_run'],'fixture-source')


if __name__=='__main__':unittest.main()
