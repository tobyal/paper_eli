"""Run with Blender --background --python this_file -- --job render_job.json.

Only imports the unchanged GLB, sets HDRI/Cycles, and exports calibrated views.
Camera records stay in the GLB Y-up frame used by trimesh/Gaussian initialization.
Depth EXRs are independently checked against GLB ray intersections before training.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import struct
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def write_png(path,pixels):
    """Store display RGB or linear mask bytes without another color transform."""
    pixels=np.ascontiguousarray(pixels,dtype=np.uint8);h,w=pixels.shape[:2]
    color=0 if pixels.ndim==2 else 2
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in pixels)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,color,0,0,0))+
                          chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    job=json.loads(Path(args.job).read_text());cfg=job['render'];out=Path(job['output'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'cameras.json').exists():
        raise RuntimeError('Refuse to overwrite a completed PBR dataset')
    for name in ('rgb','depth','mask','rgba'): (out/name).mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type=cfg.get('compute_device','CUDA');prefs.get_devices()
    # CUDA_VISIBLE_DEVICES restricts this process to the assigned shared GPU.
    chosen=[]
    for device in prefs.devices:
        device.use=device.type==prefs.compute_device_type
        if device.use: chosen.append({'id':device.id,'name':device.name,'type':device.type})
    if not chosen: raise RuntimeError('No Cycles CUDA device exposed; refusing an implicit CPU fallback')
    scene.cycles.device='GPU';scene.cycles.samples=cfg['samples']
    scene.cycles.use_adaptive_sampling=False
    scene.cycles.use_denoising=cfg.get('denoise',True)
    scene.cycles.seed=job['seed'];scene.cycles.max_bounces=cfg.get('max_bounces',8)
    scene.render.threads_mode='FIXED';scene.render.threads=8
    scene.render.resolution_x=scene.render.resolution_y=cfg['resolution']
    scene.render.resolution_percentage=100
    scene.render.film_transparent=True
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.render.image_settings.color_depth='8'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=cfg.get('exposure',0.);scene.view_settings.gamma=1
    scene.display_settings.display_device='sRGB'
    scene.render.use_file_extension=True
    bpy.ops.import_scene.gltf(filepath=job['glb'])
    meshes=[obj for obj in scene.objects if obj.type=='MESH']
    if not meshes: raise ValueError('GLB has no meshes')
    # glTF importer changes +Y-up to Blender +Z-up: (x,y,z) -> (x,-z,y).
    glb_to_blender=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    corners=[]
    for obj in meshes:
        corners.extend([np.asarray(obj.matrix_world@Vector(c))@glb_to_blender for c in obj.bound_box])
    corners=np.asarray(corners);low=corners.min(0);high=corners.max(0);center=(low+high)/2
    radius=np.linalg.norm(high-low)/2
    fov=math.radians(cfg.get('fov_degrees',49.1))
    distance=radius/math.sin(fov/2)*cfg.get('camera_margin',1.12)
    world=bpy.data.worlds.new('Factory HDRI');scene.world=world;world.use_nodes=True
    nodes=world.node_tree.nodes;nodes.clear();links=world.node_tree.links
    coord=nodes.new('ShaderNodeTexCoord');mapping=nodes.new('ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value[2]=math.radians(cfg['hdri_rotation_degrees'])
    env=nodes.new('ShaderNodeTexEnvironment');env.image=bpy.data.images.load(job['hdri'])
    bg=nodes.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=cfg['hdri_strength']
    output=nodes.new('ShaderNodeOutputWorld')
    links.new(coord.outputs['Generated'],mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'],env.inputs['Vector'])
    links.new(env.outputs['Color'],bg.inputs['Color']);links.new(bg.outputs['Background'],output.inputs['Surface'])
    camera_data=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',camera_data)
    scene.collection.objects.link(cam);scene.camera=cam
    camera_data.type='PERSP';camera_data.sensor_fit='HORIZONTAL';camera_data.sensor_width=36
    camera_data.lens=36/(2*math.tan(fov/2));camera_data.clip_start=max(radius*.001,.0001)
    camera_data.clip_end=distance+radius*10;camera_data.dof.use_dof=False
    scene.view_layers[0].use_pass_z=True
    scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear()
    layer=tree.nodes.new('CompositorNodeRLayers');depth=tree.nodes.new('CompositorNodeOutputFile')
    depth.base_path=str(out/'depth');depth.format.file_format='OPEN_EXR'
    depth.format.color_mode='BW';depth.format.color_depth='32';depth.format.exr_codec='ZIP'
    tree.links.new(layer.outputs['Depth'],depth.inputs[0])
    composite=tree.nodes.new('CompositorNodeComposite');tree.links.new(layer.outputs['Image'],composite.inputs[0])
    res=cfg['resolution'];focal=res/(2*math.tan(fov/2));views=[]
    materials=[]
    for material in bpy.data.materials:
        materials.append({'name':material.name,'nodes':[n.bl_idname for n in material.node_tree.nodes] if material.use_nodes else [],
                          'image_textures':[n.image.name for n in material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image] if material.use_nodes else []})
    start=time.monotonic()
    for elevation in cfg['elevations_degrees']:
        for azimuth in cfg['azimuths_degrees']:
            index=len(views);name=f'view_{index:03d}'
            az,el=np.deg2rad([azimuth,elevation])
            position=center+distance*np.array([math.sin(az)*math.cos(el),math.sin(el),math.cos(az)*math.cos(el)])
            forward=(center-position)/distance;right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
            down=np.cross(forward,right);R=np.stack((right,down,forward));t=-R@position
            # CV +x right,+y down,+z forward -> Blender +x right,+y up,-z forward.
            pose=np.eye(4);pose[:3,:3]=glb_to_blender@R.T@np.diag([1,-1,-1]);pose[:3,3]=glb_to_blender@position
            cam.matrix_world=Matrix(pose.tolist())
            depth.file_slots[0].path=name+'_';scene.frame_set(1)
            scene.render.filepath=str(out/'rgba'/f'{name}.png')
            bpy.ops.render.render(write_still=True)
            raw=out/'depth'/f'{name}_0001.exr';raw.rename(out/'depth'/f'{name}.exr')
            # Read the saved display PNG as raw values; composite in that same
            # display space, matching the existing Gaussian scene color pipeline.
            image=bpy.data.images.load(scene.render.filepath,check_existing=False)
            image.colorspace_settings.name='Non-Color'
            pixels=np.asarray(image.pixels[:],dtype=np.float32).reshape(res,res,4)[::-1]
            if not pixels.size: raise RuntimeError('Empty Render Result')
            alpha=pixels[:,:,3:4].clip(0,1)
            rgb=pixels[:,:,:3].clip(0,1)*alpha+1-alpha
            write_png(out/'rgb'/f'{name}.png',np.round(rgb*255).astype(np.uint8))
            write_png(out/'mask'/f'{name}.png',np.round(alpha[:,:,0]*255).astype(np.uint8))
            bpy.data.images.remove(image)
            record={'name':name,'R':R.tolist(),'t':t.tolist(),'width':res,'height':res,
                    'fx':focal,'fy':focal,'cx':res/2,'cy':res/2,
                    'azimuth_degrees':azimuth,'elevation_degrees':elevation,
                    'split':'validation' if (index%len(cfg['azimuths_degrees']))%6==5 else 'train',
                    'rgb':f'rgb/{name}.png','rgba':f'rgba/{name}.png','depth':f'depth/{name}.exr','mask':f'mask/{name}.png',
                    'blender_camera_to_world':pose.tolist()}
            views.append(record)
            write(out/'progress.json',{'completed_views':len(views),'seconds':time.monotonic()-start})
            print('PBR_VIEW_COMPLETE',name,flush=True)
    write(out/'cameras.json',{'version':1,'views':views,'glb':job['glb'],'glb_sha256':checksum(job['glb']),
        'hdri':job['hdri'],'hdri_sha256':checksum(job['hdri']),'config':cfg,'devices':chosen,
        'blender_version':bpy.app.version_string,'seconds':time.monotonic()-start,
        'local_frame':'GLB Y-up, unmodified mesh coordinates','camera_frame':'CV: x right, y down, z forward',
        'pixel_convention':'pixel centers (u+0.5,v+0.5), principal point W/2,H/2',
        'depth_definition':'camera_z (must pass independent triangle intersection calibration)',
        'rgb_color_space':'display sRGB, Standard transform, white composite in display space',
        'glb_to_blender':glb_to_blender.tolist(),'bbox_low':low.tolist(),'bbox_high':high.tolist(),
        'radius':float(radius),'camera_distance':float(distance),'materials':materials})


if __name__=='__main__': main()
