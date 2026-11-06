"""Common calibrated-camera and orbit exports for actual Gaussian outputs."""
import math
import subprocess
import numpy as np
import torch
import imageio.v2 as imageio
from .context import gsle, read_json, write_json


class VideoWriter:
    """Raw RGB pipe to system ffmpeg, independent of imageio plugin discovery."""
    def __init__(self,path,fps):
        self.path=path; self.temporary=path.with_name(path.stem+'.encoding.mp4')
        self.fps=fps; self.process=None; self.shape=None
    def __enter__(self): return self
    def append_data(self,frame):
        if self.process is None:
            h,w=frame.shape[:2]; self.shape=frame.shape
            self.process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(self.fps),
                '-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','medium','-crf','18',
                '-pix_fmt','yuv420p',str(self.temporary)],stdin=subprocess.PIPE)
        assert frame.shape==self.shape and frame.dtype==np.uint8
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
    def __exit__(self,kind,value,traceback):
        if self.process:
            self.process.stdin.close()
            code=self.process.wait()
            if code and kind is None: raise RuntimeError(f'ffmpeg failed with exit code {code}')
            if not code and kind is None: self.temporary.replace(self.path)


def render_outputs(ctx, stage, object_path, background_path=None):
    c=gsle(ctx); out=ctx.stage(stage)/'renders'; out.mkdir(exist_ok=True)
    original=c.GaussianData.load(ctx.source(ctx.cfg['project']['source_ply']))
    if background_path is None:
        from .m3_composition import effective_background
        background_path=effective_background(ctx,stage)
    background=c.GaussianData.load(background_path)
    obj=c.GaussianData.load(object_path)
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'))
    m1=read_json(ctx.stage('m1')/'manifest.json')
    center=np.asarray(m1['region_center']); up=np.asarray(m1['world_up'])
    direction=camera.center-center; radius=np.linalg.norm(direction)
    horizontal=c.normalized(direction-up*np.dot(direction,up))
    right=c.normalized(np.cross(up,horizontal))
    elevation=np.arcsin(np.dot(direction/radius,up))
    res=ctx.cfg['render']['orbit_resolution']
    count=ctx.cfg['render']['orbit_frames']; fps=ctx.cfg['render']['fps']
    def uint8(tensor): return np.round(tensor.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    with torch.no_grad():
        c.save_image(out/'anchor_before.png',c.render(camera,original,aux=False)['rgb'])
        c.save_image(out/'anchor_background.png',c.render(camera,background,aux=False)['rgb'])
        c.save_image(out/'anchor_after.png',c.render_joint(camera,background,obj,aux=False)['rgb'])
        c.save_image(out/'anchor_object.png',c.render(camera,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb'])
        orbit=[]
        with VideoWriter(out/'orbit_before_after.mp4',fps) as writer, \
             VideoWriter(out/'object_orbit.mp4',fps) as object_writer:
            for i in range(count):
                azimuth=2*math.pi*i/count
                position=center+radius*(math.cos(elevation)*(math.cos(azimuth)*horizontal+math.sin(azimuth)*right)+math.sin(elevation)*up)
                cam=c.look_at(position,center,up,res,name=f'orbit_{i:03d}')
                before=uint8(c.render(cam,original,aux=False)['rgb'])
                after=uint8(c.render_joint(cam,background,obj,aux=False)['rgb'])
                writer.append_data(np.concatenate((before,after),axis=1))
                object_writer.append_data(uint8(c.render(cam,obj,background=torch.ones(3,device='cuda'),aux=False)['rgb']))
                if i % max(1,count//8)==0:
                    imageio.imwrite(out/f'view_{i:03d}.png',np.concatenate((before,after),axis=1))
                orbit.append(cam.record())
        cameras=c.source_cameras(ctx.cfg['render']['trajectory_width'])
        with VideoWriter(out/'trajectory_before_after.mp4',fps) as writer:
            for cam in cameras:
                writer.append_data(np.concatenate((uint8(c.render(cam,original,aux=False)['rgb']),
                    uint8(c.render_joint(cam,background,obj,aux=False)['rgb'])),axis=1))
    write_json(out/'cameras.json',{'orbit':orbit,'trajectory':[cam.record() for cam in cameras],
                                'video_panels':['original','replacement'],'fps':fps})
    del original,background,obj
    torch.cuda.empty_cache()
    return out
