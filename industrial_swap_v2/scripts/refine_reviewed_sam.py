"""Apply recorded positive/negative review clicks through the existing SAM model.

Every output mask pixel is predicted by SAM; no hand-drawn contour or generated
image is substituted. This is an explicit review adapter, not automatic inference.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from industrial_swap_v2.core.context import read_json,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'repos/LangSplat'))
    from segment_anything import sam_model_registry,SamPredictor
    predictor=SamPredictor(sam_model_registry['vit_h'](checkpoint=str(root/'models/torch/hub/checkpoints/sam_vit_h_4b8939.pth')).cuda())
    items={i['key']:i for i in read_json(a.batch/'manifest.json')['items']}
    for key,record in read_json(a.prompts).items():
        item=items[key];image=np.asarray(Image.open(item['input_path']).convert('RGB'))
        out=a.batch/'sam_review_v2'/key;out.mkdir(parents=True,exist_ok=True)
        if (out/'selection.json').exists():continue
        kwargs={k:np.asarray(record[k]) for k in ('box','point_coords','point_labels')}
        predictor.set_image(image)
        with torch.no_grad():masks,scores,_=predictor.predict(**kwargs,multimask_output=True)
        candidates=[]
        board=Image.new('RGB',(1200,440),'white');draw=ImageDraw.Draw(board)
        for j,(mask,score) in enumerate(zip(masks,scores)):
            covered=mask[kwargs['point_coords'][:,1].astype(int),kwargs['point_coords'][:,0].astype(int)]
            pos=float(covered[kwargs['point_labels']==1].mean());neg=float(covered[kwargs['point_labels']==0].mean())
            valid=pos==1 and neg==0 and not(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            candidates.append({'index':j,'sam_score':float(score),'positive_coverage':pos,'negative_coverage':neg,'valid':bool(valid),'area':int(mask.sum())})
            white=image.copy();white[~mask]=255;board.paste(ImageOps.contain(Image.fromarray(white),(400,400)),(j*400,40))
            draw.text((j*400+5,5),f'{j} SAM {score:.4f} pos {pos:.2f} neg {neg:.2f}',fill='black')
        np.savez_compressed(out/'candidates.npz',**{str(i):m for i,m in enumerate(masks)})
        board.save(out/'candidates_review.jpg');write_json(out/'candidates.json',candidates)
        valid=[c for c in candidates if c['valid']]
        if not valid:raise ValueError('Recorded SAM prompts failed: '+key)
        chosen=max(valid,key=lambda c:c['sam_score']);mask=masks[chosen['index']]
        Image.fromarray(mask.astype('uint8')*255).save(out/'object_mask.png')
        Image.fromarray(np.dstack((image,mask.astype('uint8')*255))).save(out/'object_rgba.png')
        white=image.copy();white[~mask]=255;Image.fromarray(white).save(out/'object_white.png')
        write_json(out/'selection.json',{'method':'SAM ViT-H with recorded visual-review positive/negative points',
            'source_image':item['input_path'],'source_sha256':sha256(item['input_path']),
            'manual_coordinates':True,'reviewer':'assistant visual inspection of supplied image',
            'prompt_record':record,'mask_pixels_source':'unmodified SAM prediction',
            'selected':chosen,'review_status':'pending_visual_review'})
        print(key,chosen,flush=True)


if __name__=='__main__':main()
