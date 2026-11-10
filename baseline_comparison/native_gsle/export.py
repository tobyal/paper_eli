"""Export actual GSLE stages and calibrated multi-view Gaussian renders."""
import os,sys,shutil,json
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['ROOT'])/'baseline_comparison/vendor/gsle_native'))
from common import *
from PIL import ImageDraw,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def montage(name,paths,labels=None,cols=3):
    sheet=Image.new('RGB',(512*cols,540*((len(paths)+cols-1)//cols)),'white');draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(paths):
        x,y=i%cols*512,i//cols*540;draw.text((x+8,y+7),labels[i] if labels else str(p),fill='black')
        sheet.paste(ImageOps.contain(Image.open(OUT/p).convert('RGB'),(512,512)),(x,y+28))
    sheet.save(OUT/name)


def main():
    mapping={'avp_selected_anchor.png':'avp/anchor.png','avp_depth.png':'avp/depth_control.png','avp_inpainting_mask.png':'avp/bbox_mask.png',
       'inpainting_input.png':'avp/anchor.png','inpainting_mask.png':'avp/bbox_mask.png','inpainting_depth.png':'avp/depth_control.png',
       'inpainting_result.png':'inpainting/edited_anchor.png','foreground_mask.png':'foreground/mask.png',
       'foreground_rgba.png':'foreground/object_rgba.png','foreground_white.png':'foreground/reference_white.png',
       'coarse_initial.png':'coarse/native_initial.png','coarse_middle.png':'coarse/previews/reference_003500.png',
       'coarse_final.png':'coarse/previews/reference_006500.png','refine_initial.png':'refine/native_initial.png',
       'refine_middle.png':'refine/previews/reference_001000.png','refine_final.png':'refine/previews/reference_002000.png'}
    for dst,src in mapping.items():
        if (OUT/src).exists():shutil.copy2(OUT/src,OUT/dst)
    anchor=json.loads((OUT/'avp/anchor.json').read_text());cam=Camera.from_record(anchor['camera'])
    save_json(OUT/'avp_camera.json',anchor)
    target=editing_region();corners=np.asarray(target['bbox_corners'])
    from localize import box_overlay
    box_overlay(Image.open(OUT/'avp/anchor.png').convert('RGB'),cam,corners).save(OUT/'avp_anchor_with_bbox.png')
    candidates=json.loads((OUT/'avp/candidates.json').read_text())
    indices=list(range(0,len(candidates),10))
    if anchor['selected']['index'] not in indices:indices.append(anchor['selected']['index'])
    montage('avp_candidates.png',[f'avp/candidates/{i:03d}.png' for i in indices],
       [f'AVP {i}'+(' selected' if i==anchor['selected']['index'] else '') for i in indices],cols=4)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,key,label in zip(axes,['literal_abs_min_S_minus_1','text_contrast'],['Algorithm 1 score (minimize)','Prose contrast (diagnostic only)']):
        xx=[r['azimuth'] for r in candidates if r['score']];yy=[r['score'][key] for r in candidates if r['score']]
        ax.plot(xx,yy);ax.axvline(anchor['selected']['azimuth'],color='red');ax.set_ylabel(label)
    axes[-1].set_xlabel('Azimuth degrees');fig.tight_layout();fig.savefig(OUT/'avp_scores.png');plt.close(fig)
    if not (OUT/'final/scene.ply').exists():return
    scene=GaussianData.load(OUT/'final/scene.ply');original_cams=[Camera.from_record(x) for x in json.loads((OUT/'localization/review_cameras.json').read_text())]
    with torch.no_grad():
        save_image(OUT/'scene_final.png',render(cam,scene,aux=False)['rgb'])
        for i,c in enumerate(original_cams):save_image(OUT/f'final_view_{i}.png',render(c,scene,aux=False)['rgb'])
        for stage in ['coarse','refine']:
            obj=GaussianData.load(OUT/stage/'final_object_world.ply')
            for i,offset in enumerate([0,-17,17]):
                ci=(anchor['selected']['index']+offset)%100;c=Camera.from_record(candidates[ci]['camera'])
                save_image(OUT/f'{stage}/orbit_{i}.png',render(c,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'])
            montage(stage+'_multiview.png',[f'{stage}/orbit_{i}.png' for i in range(3)])
    mask=Image.open(OUT/'avp/bbox_mask.png');crop=mask.getbbox() or (0,0,512,512)
    Image.open(OUT/'scene_final.png').crop(crop).save(OUT/'scene_final_crop.png')
    montage('final_multiview.png',[f'final_view_{i}.png' for i in range(len(original_cams))])
    montage('final_comparison.png',['avp_selected_anchor.png','inpainting_result.png','scene_final.png'],['Original at AVP anchor','Native inpainting','Final composed Gaussian'])
    montage('coarse_refine_comparison.png',['foreground_white.png','coarse_final.png','refine_final.png'],['Generated foreground','Coarse 6500','DI-SDS 2000'])
    for name,src in [('final_scene.ply','final/scene.ply'),('coarse_object.ply','coarse/final_object_world.ply')]:
        if not (OUT/name).exists():(OUT/name).symlink_to(OUT/src)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,stage in enumerate(['coarse','refine']):
        values=[json.loads(line) for line in (OUT/stage/'training.jsonl').read_text().splitlines()]
        for ax,key in zip(axes[row],['rgb_mse','mask_mse']):
            ax.plot([v['step'] for v in values],[v[key] for v in values]);ax.set_title(stage+' '+key);ax.set_xlabel('Actual optimizer steps')
    fig.tight_layout();fig.savefig(OUT/'training_losses.png');plt.close(fig)
    save_json(OUT/'records/export.json',dict(coarse_steps=6500,coarse_middle=3500,refine_steps=2000,refine_middle=1000,
        anchor_camera=cam.record(),original_review_cameras=[c.record() for c in original_cams],crop=list(crop),
        original_view_definition='Post-localization original camera. final_comparison uses the AVP anchor consistently.',
        final_gaussians=len(scene.xyz),final_ply=str(OUT/'final_scene.ply'),all_images_actual=True))
    print('Exported actual GSLE stages',OUT,flush=True)


if __name__=='__main__':main()
