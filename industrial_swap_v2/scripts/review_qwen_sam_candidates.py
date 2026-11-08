"""Auditable SAM prompt sweep for manual Qwen scenes, without editing image pixels.

Uses the already installed SAM and LangSplat-compatible OpenCLIP. Prompts are
derived from the prepared edit region, with no device-specific color thresholds.
All predictions are retained; this utility does not submit M3 or call Qwen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from industrial_swap_v2.core.context import read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--keys', nargs='*')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'repos/LangSplat'))
    import preprocess as official
    from segment_anything import SamPredictor
    predictor = SamPredictor(official.sam_model_registry['vit_h'](
        checkpoint=str(root/'models/torch/hub/checkpoints/sam_vit_h_4b8939.pth')).cuda())
    pins = read_json(root/'records/gsle/models.json')
    network = official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(
        clip_model_pretrained=str(Path(pins['openclip']['path'])/'open_clip_pytorch_model.bin')))
    batch = read_json(args.batch/'manifest.json')
    source = {x['key']:x for x in read_json(batch['source_manifest'])['items']}
    for item in batch['items']:
        if args.keys and item['key'] not in args.keys:
            continue
        out = args.batch/'sam_review'/item['key']
        if (out/'selection.json').exists():
            print(item['key'], 'existing prompt sweep retained', flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        image = np.asarray(Image.open(item['input_path']).convert('RGB'))
        h,w = image.shape[:2]
        edit = np.asarray(Image.open(Path(source[item['key']]['output_directory'])/'edit_mask.png').resize((w,h),Image.Resampling.NEAREST))>127
        yy,xx = np.where(edit)
        box = np.array([xx.min(),yy.min(),xx.max(),yy.max()],dtype=np.float32)
        extent = box[2:]-box[:2]
        prompts = []
        for padding in (0,.08,.16):
            expanded = np.r_[box[:2]-extent*padding,box[2:]+extent*padding]
            expanded = np.clip(expanded,[0,0,0,0],[w-1,h-1,w-1,h-1])
            prompts.append({'box':expanded.tolist(),'padding':padding})
        for fy in (.30,.50,.70):
            for fx in (.30,.50,.70):
                point = box[:2]+extent*[fx,fy]
                if not edit[int(point[1]),int(point[0])]:
                    continue
                prompts.append({'point_coords':[point.tolist()],'point_labels':[1]})
                prompts.append({'box':box.tolist(),'point_coords':[point.tolist()],'point_labels':[1]})
        predictor.set_image(image)
        network.set_positives([item['description']])
        masks=[]; records=[]; seen=set()
        for prompt in prompts:
            kwargs={k:np.array(v) for k,v in prompt.items() if k in ('box','point_coords','point_labels')}
            with torch.no_grad():
                predictions,confidence,_ = predictor.predict(**kwargs,multimask_output=True)
            for mask, conf in zip(predictions,confidence):
                digest=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                if digest in seen: continue
                seen.add(digest)
                y,x=np.where(mask)
                if len(x)<max(500,edit.sum()*.1): continue
                clipped=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                intersection=int((mask&edit).sum())
                containment=intersection/max(1,int(mask.sum()))
                coverage=intersection/max(1,int(edit.sum()))
                crop=image[y.min():y.max()+1,x.min():x.max()+1].copy()
                crop[~mask[y.min():y.max()+1,x.min():x.max()+1]]=0
                crop=cv2.resize(official.pad_img(crop),(224,224))
                with torch.no_grad():
                    embedding=network.encode_image(torch.from_numpy(crop.copy()).permute(2,0,1).cuda().float()[None]/255)
                    embedding/=embedding.norm(dim=-1,keepdim=True)
                    relevance=float(network.get_relevancy(embedding,0)[0,0])
                f1=2*containment*coverage/max(1e-8,containment+coverage)
                valid=not clipped and containment>.7 and float(conf)>.85
                score=relevance*max(0,float(conf))**3*np.sqrt(f1)
                records.append({'index':len(masks),'sam_score':float(conf),'clip_score':relevance,
                    'area':int(mask.sum()),'containment':containment,'edit_coverage':coverage,
                    'image_border_clipped':clipped,'valid':valid,'score':float(score),'prompt':prompt,
                    'sha256_packed_mask':digest})
                masks.append(mask)
        valid=sorted([r for r in records if r['valid']],key=lambda r:r['score'],reverse=True)
        np.savez_compressed(out/'candidates.npz',**{str(i):m for i,m in enumerate(masks)})
        write_json(out/'candidates.json',records)
        if not valid:
            print(item['key'],'NO VALID CANDIDATES',flush=True)
            continue
        chosen=valid[0];mask=masks[chosen['index']]
        Image.fromarray(mask.astype('uint8')*255).save(out/'object_mask.png')
        Image.fromarray(np.dstack((image,mask.astype('uint8')*255))).save(out/'object_rgba.png')
        white=image.copy();white[~mask]=255;Image.fromarray(white).save(out/'object_white.png')
        write_json(out/'selection.json',{'method':'SAM automatic region-derived prompt sweep + OpenCLIP ranking',
            'source_image':item['input_path'],'source_sha256':sha256(item['input_path']),
            'mask_pixels_source':'unmodified SAM prediction','manual_coordinates':False,
            'ranking':'CLIP relevance * SAM confidence^3 * sqrt(edit-region overlap F1)',
            'selected':chosen,'candidate_count':len(records),'review_status':'pending_visual_review'})
        board=Image.new('RGB',(1600,1320),'white');draw=ImageDraw.Draw(board)
        for j,record in enumerate(valid[:12]):
            rgba=image.copy();rgba[~masks[record['index']]]=255
            col=j%4*400;row=j//4*440
            board.paste(ImageOps.contain(Image.fromarray(rgba),(400,400)),(col,row+40))
            draw.text((col+5,row+4),f"rank{j+1} id{record['index']} score{record['score']:.3f} SAM{record['sam_score']:.3f}",fill='black')
            draw.text((col+5,row+20),f"CLIP{record['clip_score']:.3f} area{record['area']}",fill='black')
        board.save(out/'candidates_review.jpg')
        print(item['key'],json.dumps(chosen),flush=True)


if __name__=='__main__':
    main()
