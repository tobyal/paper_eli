"""Small CPU tests for independent-mask scoring and missing-data safeguards."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from summarize_f08_ten import evaluate_mask,sha

class MaskEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
        Image.new('RGB',(4,4),'gray').save(self.d/'image.png')
        gt=np.zeros((4,4),np.uint8);gt[:2,:2]=255
        Image.fromarray(gt).save(self.d/'mask.png');Image.fromarray(gt).save(self.d/'photo.png')
        pred=np.zeros_like(gt);pred[2:,2:]=255;Image.fromarray(pred).save(self.d/'text.png')
        self.task=dict(task_id='unit',image_sha256=sha(self.d/'image.png'),review_path=str(self.d/'review.json'),mask_path=str(self.d/'mask.png'),
            camera=dict(width=4,height=4,name='heldout.jpg'),input_camera='reference.jpg',
            predicted_masks=dict(text_only=str(self.d/'text.png'),photo_assisted=str(self.d/'photo.png')))
        self.review=dict(task_id='unit',image_sha256=self.task['image_sha256'],reviewed=True,reviewer='unit-test',reviewer_type='human',
            visible=True,mask_origin='independent_manual',derived_from_method_prediction=False)
    def tearDown(self):self.tmp.cleanup()
    def write(self):Path(self.task['review_path']).write_text(json.dumps(self.review))
    def test_missing_not_zero(self):self.assertEqual(evaluate_mask(self.task),('pending_independent_human_annotation',None))
    def test_real_zero_and_one(self):
        self.write();status,result=evaluate_mask(self.task)
        self.assertEqual(status,'valid');self.assertEqual(result,dict(text_only=0.,photo_assisted=1.))
    def test_prediction_rejected(self):
        self.review['derived_from_method_prediction']=True;self.write()
        self.assertEqual(evaluate_mask(self.task),('mask_independence_unconfirmed',None))
    def test_hidden_excluded(self):
        self.review['visible']=False;self.write();self.assertEqual(evaluate_mask(self.task),('not_visible',None))
    def test_input_view_rejected(self):
        self.write();self.task['input_camera']='heldout.jpg'
        with self.assertRaisesRegex(ValueError,'cannot be held out'):evaluate_mask(self.task)
    def test_image_mismatch_rejected(self):
        self.review['image_sha256']='wrong';self.write()
        with self.assertRaisesRegex(ValueError,'identity mismatch'):evaluate_mask(self.task)

if __name__=='__main__':unittest.main()
