import gc,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256

def main():
    d=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F14_background_refinement';(d/'panels').mkdir(exist_ok=True)
    row=read_json(ROOT/'baseline_comparison/prompts.json')['cases'][0];base=Path(row['ours_source']);m4=base/'m4'
    ctx=Context(base/'config.yaml');c=gsle(ctx);torch.cuda.set_per_process_memory_fraction(.35)
    manifest=read_json(m4/'multiview_supervision/manifest.json');view=manifest['views'][0]
    cache=np.load(m4/'multiview_supervision/cache'/f"{view['name']}.npz")
    cam=c.Camera.from_record(view['camera']).resize(1024,1024)
    for name in ['old_mask','new_mask','environment_mask']:
        Image.fromarray((cache[name]*255).astype(np.uint8)).save(d/'panels'/(name+'.png'))
    paths={'step0':m4/'scene_initial.ply','rgb_only':d/'rgb_only_run/m4/snapshots/iteration_002000/scene.ply','full_candidate':m4/'scene_candidate_step2000.ply','guarded_final':m4/'scene_final.ply'}
    renders={};records={}
    with torch.no_grad():
        for name,p in paths.items():
            data=c.GaussianData.load(p);rgb=c.render(cam,data,aux=False)['rgb'];c.save_image(d/'panels'/(name+'.png'),rgb)
            renders[name]=rgb.cpu().numpy().transpose(1,2,0);records[name]={'path':str(p),'sha256':sha256(p)}
            del data,rgb;gc.collect();torch.cuda.empty_cache()
    env=np.asarray(Image.fromarray((cache['environment_mask']*255).astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST))>0
    heatmaps={}
    for name in ['rgb_only','full_candidate','guarded_final']:
        difference=np.abs(renders[name]-renders['step0']).mean(2);heatmaps[name]=difference*env
    np.savez_compressed(d/'background_differences.npz',environment=env,**heatmaps)
    mask=np.asarray(Image.fromarray((cache['new_mask']*255).astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST))>0
    yy,xx=np.nonzero(mask);b=[max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1024,int(xx.max())+30),min(1024,int(yy.max())+30)]
    for name in paths:Image.open(d/'panels'/(name+'.png')).crop(b).save(d/'panels'/(name+'_crop.png'))
    write_json(d/'render_sources.json',dict(camera=cam.record(),crop=b,scenes=records,mask_manifest=str(m4/'multiview_supervision/manifest.json'),same_fixed_object_sha256=sha256(m4/'object_world.ply'),mask_definition='environment = 1 - union(old,new)',no_scene_parameters_changed_for_render=True))

if __name__=='__main__':main()
