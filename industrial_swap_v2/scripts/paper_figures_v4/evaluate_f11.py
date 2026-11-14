"""Shared ROI-prompted SAM assessment; no containment-based candidate filtering."""
import sys,json
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import read_json,write_json,sha256
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F11_m2_ablation'

def main():
    sys.path.insert(0,str(ROOT/'repos/LangSplat'));import preprocess as official
    from segment_anything import SamPredictor
    sam=official.sam_model_registry['vit_h'](checkpoint=str(ROOT/'models/torch/hub/checkpoints/sam_vit_h_4b8939.pth')).cuda()
    predictor=SamPredictor(sam);pins=read_json(ROOT/'records/gsle/models.json')
    text='A yellow six-axis industrial robotic arm mounted on a square steel base, with articulated rotary joints, shoulder, elbow, wrist, and a compact two-finger gripper.'
    clip=official.OpenCLIPNetwork(official.OpenCLIPNetworkConfig(clip_model_pretrained=str(Path(pins['openclip']['path'])/'open_clip_pytorch_model.bin'),positives=(text,)))
    rows=[];provenance=[]
    for key in ['C1','C2','C3','C4']:
        d=OUT/'conditions'/key/'m2';image=np.array(Image.open(d/'edited_scene.png').convert('RGB'));h,w=image.shape[:2]
        anchor=np.array(Image.open(d/'anchor.png').convert('RGB').resize((w,h),Image.Resampling.LANCZOS))
        edit=np.array(Image.open(d/'edit_mask.png').resize((w,h),Image.Resampling.NEAREST))>127
        yy,xx=np.nonzero(edit);box=np.array([xx.min(),yy.min(),xx.max(),yy.max()]);predictor.set_image(image)
        masks,scores,_=predictor.predict(box=box,multimask_output=True);candidates=[]
        for i,(mask,confidence) in enumerate(zip(masks,scores)):
            y,x=np.nonzero(mask)
            if len(x)<20:continue
            foreground=image.copy();foreground[~mask]=0
            crop=cv2.resize(official.pad_img(foreground[y.min():y.max()+1,x.min():x.max()+1]),(224,224))
            with torch.no_grad():
                embedding=clip.encode_image(torch.from_numpy(crop.copy()).permute(2,0,1).cuda().float()[None]/255)
                embedding/=embedding.norm(dim=-1,keepdim=True);relevance=float(clip.get_relevancy(embedding,0)[0,0])
            candidates.append(dict(index=i,sam_score=float(confidence),clip_score=relevance,selection_score=float(confidence)*relevance))
        chosen=max(candidates,key=lambda x:x['selection_score']);mask=masks[chosen['index']]
        Image.fromarray(mask.astype(np.uint8)*255).save(d/'evaluation_object_mask.png')
        difference=np.abs(image.astype(float)-anchor.astype(float)).mean(2)/255;heat=difference*(~edit)
        np.save(d/'outside_edit_difference.npy',heat)
        placement=float((mask&edit).sum()/max(mask.sum(),1));iou=float((mask&edit).sum()/max((mask|edit).sum(),1))
        rows.append(dict(condition=key,object_inside_edit_fraction=placement,object_edit_iou=iou,background_rgb_mae=float(difference[~edit].mean()),
            background_psnr=float(-10*np.log10(max(np.mean(((image.astype(float)-anchor)/255)[~edit]**2),1e-12))),
            background_changed_fraction_005=float((difference[~edit]>.05).mean()),clip_text_relevance=chosen['clip_score'],
            clip_is_not_attribute_accuracy=True,object_pixels=int(mask.sum()),edit_pixels=int(edit.sum()),seed=42,samples=1))
        write_json(d/'evaluation_segmentation.json',dict(candidates=candidates,selected=chosen,box=box.tolist(),
            policy='Shared edit-envelope box, identical SAM/CLIP model. Selection by SAM confidence times CLIP relevance, without containment filtering. This is an ROI-assisted estimate, not a manually annotated mask.'))
        provenance.append(dict(condition=key,image_sha256=sha256(d/'edited_scene.png'),request=read_json(d/'api_request.json'),
            reference_mask_sha256=sha256(d/'edit_mask.png'),evaluation_segmentation=read_json(d/'evaluation_segmentation.json')))
    write_json(OUT/'metrics.json',rows);write_json(OUT/'metric_sources.json',provenance);print('F11 METRICS COMPLETE',flush=True)

if __name__=='__main__':main()
