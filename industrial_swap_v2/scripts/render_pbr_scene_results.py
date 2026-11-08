"""Read-only calibrated and orbit previews of PBR case00 and guarded M4 outputs."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
from industrial_swap_v2.core.scene_integration import object_path
from industrial_swap_v2.core.rendering import VideoWriter


def run(ctx):
    c=gsle(ctx);m4=ctx.stage('m4');report=read_json(m4/'summary.json');out=ctx.out/'previews';out.mkdir(exist_ok=True)
    paths={'object':object_path(ctx),'background_initial':ctx.stage('m1')/'background_gaussians.ply',
        'background_actual':Path(report['candidate_background']),'background_guarded':Path(report['guarded_background'])}
    data={k:c.GaussianData.load(path) for k,path in paths.items()}
    anchor=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    def array(t): return np.round(t.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    @torch.no_grad()
    def panels(camera,labels=True):
        images=[array(c.render(camera,data['object'],background=torch.ones(3,device='cuda'),aux=False)['rgb'])]
        for key in ('background_initial','background_actual','background_guarded'):
            images.append(array(c.render_joint(camera,data[key],data['object'],aux=False)['rgb']))
        content=np.concatenate(images,axis=1)
        if not labels: return content
        board=Image.new('RGB',(content.shape[1],content.shape[0]+40),'white');board.paste(Image.fromarray(content),(0,40))
        draw=ImageDraw.Draw(board)
        for i,label in enumerate(('PBR-trained fixed object','M4 insertion / before optimization',
            f'M4 actual {report["checks"]["training_steps"]} / guard '+('PASS' if report['candidate_guard_passed'] else 'FAIL'),
            f'Guarded final / selected {report["selected_step"]}')):
            draw.text((i*camera.width+8,10),label,fill='black')
        return np.asarray(board)
    Image.fromarray(panels(anchor.resize(640,round(anchor.height*640/anchor.width)))).save(out/'anchor_stages.jpg',quality=94)
    mask_manifest=read_json(m4/'multiview_supervision/manifest.json')
    val=[r for r in mask_manifest['views'] if r['split']=='validation']
    for rec in [val[i] for i in np.linspace(0,len(val)-1,4).astype(int)]:
        camera=c.Camera.from_record(rec['camera']);Image.fromarray(panels(camera)).save(out/(rec['name']+'.jpg'),quality=93)
    center=np.asarray(read_json(ctx.stage('m1')/'manifest.json')['region_center'])
    up=np.asarray(read_json(ctx.stage('m1')/'manifest.json')['world_up'])
    direction=anchor.center-center;radius=np.linalg.norm(direction);elevation=np.arcsin(np.dot(direction/radius,up))
    forward=c.normalized(direction-up*np.dot(direction,up));right=c.normalized(np.cross(up,forward))
    orbit=[]
    with VideoWriter(out/'orbit_insertion_actual_guarded.mp4',12) as writer:
        for i in range(72):
            az=2*np.pi*i/72;pos=center+radius*(math.cos(elevation)*(math.cos(az)*forward+math.sin(az)*right)+math.sin(elevation)*up)
            cam=c.look_at(pos,center,up,512,name=f'orbit_{i:03d}');writer.append_data(panels(cam));orbit.append(cam.record())
    source=c.source_cameras(640)
    selected=[source[i] for i in np.linspace(0,len(source)-1,min(72,len(source))).astype(int)]
    with VideoWriter(out/'trajectory_insertion_actual_guarded.mp4',12) as writer:
        for cam in selected: writer.append_data(panels(cam))
    write_json(out/'render_receipt.json',{'inputs':{k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()},
        'panels':['fixed_object','initial_insertion','actual_training_candidate','guarded_final'],
        'orbit':orbit,'trajectory':[cam.record() for cam in selected],
        'actual_step':report['checks']['training_steps'],'selected_step':report['selected_step'],
        'candidate_guard_passed':report['candidate_guard_passed']})
    print('PBR scene previews completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();run(Context(args.config))
