import unittest
import numpy as np
from industrial_swap_v2.core.m1_reference import selected_region,rank_descriptions,region_proposals


class OriginalPhotoFeatures(unittest.TestCase):
    def setUp(self):
        self.segments=np.full((4,4,6),-1,dtype=np.int32)
        self.segments[3,:2,:3]=0;self.segments[3,2:,:3]=1;self.segments[2,:,3:]=2
        self.features=np.zeros((3,512),dtype=np.float32)
        self.features[0,0]=2;self.features[1,1]=3;self.features[2,2]=1

    def test_merge_uses_disjoint_region_areas_and_normalized_cached_features(self):
        mask,raw,weights,prototype=selected_region(self.segments,self.features,3,[1,0,1])
        self.assertEqual(mask.sum(),12);np.testing.assert_array_equal(raw,self.features[:2])
        np.testing.assert_allclose(weights,[.5,.5]);np.testing.assert_allclose(prototype[:2],np.sqrt(.5))
        with self.assertRaises(ValueError):selected_region(self.segments,self.features,3,[2])

    def test_ranking_changes_with_selected_image_feature(self):
        embeddings=np.eye(512,dtype=np.float32)[:3];negatives=-np.ones((4,512),dtype=np.float32)/np.sqrt(512)
        texts=['cabinet','cooler','rack']
        a=rank_descriptions(embeddings[0],texts,embeddings,negatives)
        b=rank_descriptions(embeddings[1],texts,embeddings,negatives)
        self.assertEqual(a[0]['text'],'cabinet');self.assertEqual(b[0]['text'],'cooler')
        self.assertAlmostEqual(b[0]['cosine_similarity'],1)

    def test_box_proposes_cached_mask_not_box_segmentation(self):
        proposals=region_proposals(self.segments,3,box=[0,0,2,2])
        self.assertEqual(proposals[0]['id'],0);self.assertEqual(proposals[0]['pixels'],6)
        self.assertEqual(proposals[0]['bbox'],[0,0,3,2])
        self.assertEqual(region_proposals(self.segments,3,point=[5,3]),[])


if __name__=='__main__':unittest.main()
