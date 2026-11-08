"""Render verified M3/M4 results to a separate deliverable directory, no training."""
import argparse
from pathlib import Path
import shutil
from contextlib import nullcontext
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
from industrial_swap_v2.core.context import Context,read_json,write_json,sha256,gsle
from industrial_swap_v2.core.rendering import VideoWriter


def uint8(value):
    return np.rint(value.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--keys',nargs='*');a=p.parse_args()
    batch=read_json(a.batch/'final_manifest.json')
    for item in batch['items']:
        if a.keys and item['key'] not in a.keys:continue
        states=item['states']
        if any(states.get(s,{}).get('status')!='completed' for s in ('m2','m3','m4')):continue
        out=a.batch/'results'/item.get('result_subdir',item['key']);out.mkdir(parents=True,exist_ok=True)
        if (out/'render_receipt.json').exists() and (out/'candidate_video_receipt.json').exists():continue
        ctx=Context(Path(states['m4']['folder'])/'config.yaml');c=gsle(ctx)
        m2=Path(states['m2']['output']);m3=Path(states['m3']['output']);m4=Path(states['m4']['output'])
        summary=read_json(m4/'summary.json');background=Path(summary['guarded_background'])
        objpath=m3/'object_world.ply';initial=ctx.stage('m1')/'background_gaussians.ply'
        assert sha256(objpath)==summary['candidate_sha256']['object']
        assert sha256(background)==read_json(m4/'background_optimization/summary.json')['background_sha256']
        original=c.GaussianData.load(ctx.source(ctx.cfg['project']['source_ply']))
        bg0=c.GaussianData.load(initial);bg1=c.GaussianData.load(background);obj=c.GaussianData.load(objpath)
        actual=c.GaussianData.load(summary['candidate_background'])
        final_is_initial=sha256(background)==sha256(initial)
        camera=c.Camera.from_record(read_json(m2/'anchor_camera.json'))
        with torch.no_grad():
            values={'original_scene':c.render(camera,original,aux=False)['rgb'],
                    'object_alone':c.render(camera,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'],
                    'm3_scene':c.render_joint(camera,bg0,obj,aux=False)['rgb'],
                    'm4_final':c.render_joint(camera,bg1,obj,aux=False)['rgb']}
            for name,value in values.items():Image.fromarray(uint8(value)).save(out/(name+'.png'))
            inputs=[Image.open(item['input_path']).convert('RGB')]+[Image.fromarray(uint8(values[k])) for k in ('object_alone','m3_scene','m4_final')]
            labels=['User Qwen image','M3 fixed device alone','M3 original M1 background',f'M4 final (selected step {summary["selected_step"]})']
            board=Image.new('RGB',(1920,524),'white');draw=ImageDraw.Draw(board)
            for j,(im,label) in enumerate(zip(inputs,labels)):
                board.paste(ImageOps.contain(im,(480,480)),(j*480,44));draw.text((j*480+8,12),label,fill='black')
            board.save(out/'stages_same_camera.jpg')
            cameras=read_json(m3/'renders/cameras.json')
            for kind in ('orbit','trajectory'):
                guarded_path=out/(kind+'_m3_m4.mp4')
                with (nullcontext(None) if guarded_path.exists() else VideoWriter(guarded_path,ctx.cfg['render']['fps'])) as writer, \
                     VideoWriter(out/(kind+'_m3_actual250_final.mp4'),ctx.cfg['render']['fps']) as actual_writer:
                    for record in cameras[kind]:
                        cam=c.Camera.from_record(record)
                        before=uint8(c.render_joint(cam,bg0,obj,aux=False)['rgb'])
                        after=before if final_is_initial else uint8(c.render_joint(cam,bg1,obj,aux=False)['rgb'])
                        candidate=uint8(c.render_joint(cam,actual,obj,aux=False)['rgb'])
                        if writer:writer.append_data(np.concatenate((before,after),axis=1))
                        panel=Image.new('RGB',(before.shape[1]*3,before.shape[0]+32),'white');draw=ImageDraw.Draw(panel)
                        for j,(pixels,label) in enumerate(zip((before,candidate,after),('M3 input','M4 ACTUAL step 250',f'M4 GUARDED selected step {summary["selected_step"]}'))):
                            panel.paste(Image.fromarray(pixels),(j*before.shape[1],32));draw.text((j*before.shape[1]+8,8),label,fill='black')
                        actual_writer.append_data(np.asarray(panel))
            del values
        for name,source in [('object_world.ply',objpath),('scene_final.ply',m4/'scene_final.ply'),
                            ('background_optimized.ply',background),('trellis_asset.glb',m3/'trellis_asset.glb')]:
            path=out/name
            if not path.exists():path.symlink_to(source.resolve())
        shutil.copy2(m4/'summary.json',out/'m4_report.json')
        shutil.copy2(m4/'validation/supervision_union_masks.png',out/'supervision_union_masks.png')
        write_json(out/'render_receipt.json',{'key':item['key'],'selected_step':summary['selected_step'],
            'candidate_guard_passed':summary['candidate_guard_passed'],
            'background_sha256':sha256(background),'object_sha256':sha256(objpath),
            'scene_final_sha256':sha256(m4/'scene_final.ply'),'camera':camera.record(),
            'video_panels':['M3 original M1 background + fixed new device','M4 guarded background + identical new device'],
            'orbit_frames':len(cameras['orbit']),'trajectory_frames':len(cameras['trajectory']),
            'original_experiment_outputs_modified':False})
        write_json(out/'candidate_video_receipt.json',{'object_sha256':sha256(objpath),
            'candidate_background_sha256':sha256(summary['candidate_background']),
            'panels':['M3 input','M4 actual step 250','M4 guarded final'],
            'selected_step':summary['selected_step'],'candidate_guard_passed':summary['candidate_guard_passed'],
            'orbit_frames':len(cameras['orbit']),'trajectory_frames':len(cameras['trajectory'])})
        del original,bg0,bg1,obj,actual;torch.cuda.empty_cache()
        print(item['key'],'final renders exported',flush=True)


if __name__=='__main__':main()
