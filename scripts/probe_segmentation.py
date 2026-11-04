"""Inspect the official LangSAM masks before using them for editing."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from lang_sam import LangSAM

root=Path(os.environ['ROOT'])
destination=root/'outputs/segmentation_probe'
destination.mkdir(parents=True,exist_ok=True)
model=LangSAM('vit_h')
records=[]
for index in [61,91,111,151,161,191,241]:
    subset='test' if (index-1)%8==0 else 'train'
    source=root/f'data/edit_{subset}/images/frame_{index:06d}.jpg'
    image=Image.open(source).convert('RGB')
    for label,prompt in [('housing','white industrial machine housing'),('machine','industrial machine'),('base','white machine base'),('cart','white cart'),('platform','white platform'),('trolley','white trolley')]:
        with torch.no_grad():
            masks,boxes,phrases,logits=model.predict(image,prompt)
        overlay=np.array(image).astype(np.float32)
        if len(masks):
            mask=masks[0].numpy().astype(bool)
            overlay[mask]=0.5*overlay[mask]+np.array([127.5,0,0])
            Image.fromarray((mask*255).astype(np.uint8)).save(destination/f'frame_{index:06d}_{label}_mask.png')
        output=Image.fromarray(overlay.astype(np.uint8))
        draw=ImageDraw.Draw(output)
        for i,box in enumerate(boxes):
            draw.rectangle(box.tolist(),outline='yellow' if i==0 else 'cyan',width=2)
        output.save(destination/f'frame_{index:06d}_{label}_overlay.jpg')
        records.append({'frame':index,'prompt':prompt,'detections':len(masks),'boxes':boxes.tolist(),'phrases':phrases,'logits':logits.tolist(),'first_mask_fraction':float(masks[0].float().mean()) if len(masks) else 0})
(destination/'detections.json').write_text(json.dumps(records,indent=2))
