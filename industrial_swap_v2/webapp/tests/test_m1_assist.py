"""No GPU/network/model execution; isolated photo/cache and durable M1 queue tests."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[3]
os.environ['SWAP_WEB_DATA']=str(ROOT/'industrial_swap_v2/webapp/data'/('test_m1_'+uuid.uuid4().hex))
from industrial_swap_v2.webapp import store as s,m1_cases as m
from industrial_swap_v2.webapp.app import app
from industrial_swap_v2.core.context import read_json


class M1Web(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app,headers={'X-Swap-Action':'1'})
        self.fixture=s.DATA/'tmp'/s.uid();self.fixture.mkdir()
        self.photo=self.fixture/'photo.jpg';Image.new('RGB',(6,4),(110,130,150)).save(self.photo)
        self.sp=self.fixture/'sam.npy';self.fp=self.fixture/'clip.npy'
        self.segments=np.full((4,4,6),-1,dtype='int32');self.segments[3,:,:3]=0;self.segments[3,:,3:]=1
        self.features=np.eye(512,dtype='float32')[:6];np.save(self.sp,self.segments);np.save(self.fp,self.features)
        self.camera={'R':np.eye(3).tolist(),'t':[0,0,1],'width':6,'height':4,'fx':4,'fy':4,'cx':3,'cy':2,'name':'fixture.jpg'}
        (m.HOME/'vocabulary').mkdir(exist_ok=True);(m.HOME/'vocabulary/text_embeddings.npz').touch()
        cache={'texts':['a cabinet','a cooler','a rack','a fan','a press','a conveyor'],'embeddings':self.features,'negatives':-self.features[:4]}
        patches=[patch.object(m.ref,'frame_arrays',return_value=(self.segments,self.features)),
                 patch.object(m.ref,'frame_paths',return_value=(self.photo,self.sp,self.fp)),
                 patch.object(m.ref,'cameras',return_value={'fixture.jpg':self.camera}),
                 patch.object(m,'text_cache',return_value=(cache,{'model':'synthetic unit fixture'}))]
        for item in patches:item.start();self.addCleanup(item.stop)

    def select(self,index=0):
        response=self.client.post('/api/m1-assist/select',json={'frame':'fixture.jpg','level':3,'ids':[index]})
        self.assertEqual(response.status_code,200,response.text);return response.json()

    def test_region_rank_and_saved_original_feature(self):
        first=self.select();second=self.select(1)
        self.assertEqual(first['top5'][0]['text'],'a cabinet');self.assertEqual(second['top5'][0]['text'],'a cooler')
        self.assertEqual(len(first['top5']),5)
        path=m.selection_path(first['id']);self.assertEqual((path/'original_photo.jpg').read_bytes(),self.photo.read_bytes())
        np.testing.assert_array_equal(np.load(path/'clip_region_features.npy'),self.features[:1])
        self.assertNotEqual(first['id'],second['id'])

    def test_localization_queue_is_m1_only_immutable_and_idempotent(self):
        selected=self.select();before=len(s.rows('SELECT id FROM stage_runs'))
        payload={'selection_id':selected['id'],'text':selected['top5'][0]['text'],'request_id':s.uid(),'validation':True}
        first=self.client.post('/api/m1-assist/localize',json=payload);self.assertEqual(first.status_code,200,first.text)
        again=self.client.post('/api/m1-assist/localize',json=payload);self.assertEqual(first.json()['id'],again.json()['id'])
        row=m.job(first.json()['id']);folder=Path(row['folder']);job=read_json(folder/'job.json')
        self.assertEqual(job['stages'],['m1']);self.assertEqual(job['selection'],str(folder/'reference'))
        self.assertEqual(len(s.rows('SELECT id FROM stage_runs')),before)
        self.assertEqual(self.client.post(f"/api/m1-assist/jobs/{row['id']}/confirm",json={'name':'premature'}).status_code,422)
        payload['text']='a different device'
        self.assertEqual(self.client.post('/api/m1-assist/localize',json=payload).status_code,422)
        self.assertFalse(any(c['job_id']==row['id'] for c in self.client.get('/api/m1-assist/cases').json()))

    def test_invalid_scale_coordinates_and_missing_cache_fail_closed(self):
        self.assertEqual(self.client.post('/api/m1-assist/regions',json={'frame':'fixture.jpg','level':3,'point':[9,2]}).status_code,422)
        self.assertEqual(self.client.post('/api/m1-assist/regions',json={'frame':'fixture.jpg','level':3,'point':[1.1,2]}).status_code,422)
        self.assertEqual(self.client.post('/api/m1-assist/select',json={'frame':'fixture.jpg','level':2,'ids':[0]}).status_code,422)


if __name__=='__main__':unittest.main()
