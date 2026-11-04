"""Annotate the evaluation target with SAM; not an editing algorithm."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from segment_anything import sam_model_registry,SamPredictor

root=Path(os.environ['ROOT'])
destination=root/'outputs/reference_sam'
destination.mkdir(parents=True,exist_ok=True)
image=np.array(Image.open(root/'data/edit_train/images/frame_000151.jpg').convert('RGB'))
model=sam_model_registry['vit_h'](checkpoint=str(root/'models/torch/hub/checkpoints/sam_vit_h_4b8939.pth')).cuda()
predictor=SamPredictor(model)
predictor.set_image(image)
box=np.array([145,260,292,359])
points=np.array([[215,310]])
masks,scores,logits=predictor.predict(point_coords=points,point_labels=np.ones(1),box=box,multimask_output=True)
for i,mask in enumerate(masks):
    Image.fromarray((mask*255).astype(np.uint8)).save(destination/f'mask_{i}.png')
    overlay=image.astype(np.float32)
    overlay[mask]=overlay[mask]*.5+np.array([127.5,0,0])
    preview=Image.fromarray(overlay.astype(np.uint8))
    ImageDraw.Draw(preview).rectangle(box.tolist(),outline='yellow',width=1)
    preview.save(destination/f'overlay_{i}.jpg')
(destination/'annotation.json').write_text(json.dumps({'reference':'frame_000151.jpg','box':box.tolist(),'positive_points':points.tolist(),'scores':scores.tolist(),'area':[int(mask.sum()) for mask in masks]},indent=2))
print((destination/'annotation.json').read_text())
