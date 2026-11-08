"""Official TRELLIS.2 inference in its isolated Python/CUDA environment."""
import os
import sys
import time
import subprocess
from pathlib import Path
from .context import Context, arguments, read_json, write_json, sha256


def run(ctx):
    os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                      HF_HOME=str(ctx.path('models/huggingface')),HF_HUB_OFFLINE='1')
    repo=ctx.path('third_party/TRELLIS.2')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit==ctx.cfg['m3']['revision'], 'Unexpected official TRELLIS code revision'
    sys.path.insert(0,str(repo))
    import torch
    import numpy as np
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.representations import MeshWithVoxel
    import o_voxel
    out=ctx.stage('m3'); cfg=ctx.cfg['m3']; input_path=ctx.path(cfg['input_image'])
    if not input_path.is_file(): raise FileNotFoundError('M2 foreground must exist before TRELLIS inference')
    if torch.cuda.mem_get_info()[0]/2**30 < cfg['minimum_free_gpu_gib']:
        raise RuntimeError('Insufficient free GPU memory for official TRELLIS; other jobs are untouched')
    if cfg.get('memory_budget_gib'):
        torch.cuda.set_per_process_memory_fraction(cfg['memory_budget_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    write_json(out/'resource_policy.json',{'physical_gpu':torch.cuda.get_device_name(0),
        'available_gib_at_start':torch.cuda.mem_get_info()[0]/2**30,
        'torch_allocator_budget_gib':cfg.get('memory_budget_gib'),'official_low_vram':cfg['low_vram'],
        'pipeline_type':cfg['pipeline_type'],'note':'Resource-only adaptation; official samplers, resolutions and weights unchanged. CUDA library workspace may exist outside Torch allocator.'})
    fingerprint={'input_sha256':sha256(input_path),'repository_commit':commit,
                 'model':cfg['model'],'seed':ctx.cfg['project']['seed'],'pipeline_type':cfg['pipeline_type']}
    record=out/'trellis_generation.json'; raw=out/'trellis_native.pt'; glb_path=out/'trellis_asset.glb'
    if record.exists():
        old=read_json(record)
        assert all(old[k]==v for k,v in fingerprint.items()), 'Saved TRELLIS input/config mismatch'
    start=time.time()
    if raw.exists():
        mesh=MeshWithVoxel(**{k:v.cuda() if isinstance(v,torch.Tensor) else v
                            for k,v in torch.load(raw,map_location='cpu',weights_only=False).items()})
    else:
        pin=read_json(ctx.records/'trellis_model.json')
        deps=read_json(ctx.records/'trellis_dependencies.json')
        if not pin.get('download_complete') or any(v.get('status')!='complete' for v in deps.values()):
            raise RuntimeError('Official model downloads are not complete')
        config=pin['pipeline']
        for name,path in config['args']['models'].items():
            if path.startswith('microsoft/TRELLIS-image-large/'):
                path=str(Path(deps['microsoft/TRELLIS-image-large']['snapshot_path'])/path.split('/',2)[2])
            else:
                path=str(Path(pin['snapshot_path'])/path)
            config['args']['models'][name]=path
        for name in ('image_cond_model','rembg_model'):
            model=config['args'][name]['args']['model_name']
            config['args'][name]['args']['model_name']=deps[model]['snapshot_path']
        local=ctx.path('models/pinned_pipeline'); local.mkdir(exist_ok=True)
        write_json(local/'pipeline.json',config)
        pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(local))
        pipeline.low_vram=cfg['low_vram']; pipeline.cuda()
        with Image.open(input_path) as im: image=im.convert('RGBA')
        pipeline.preprocess_image(image).save(out/'trellis_condition.png')
        mesh=pipeline.run(image,seed=ctx.cfg['project']['seed'],pipeline_type=cfg['pipeline_type'])[0]
        mesh.simplify(16777216)
        state={k:getattr(mesh,k) for k in ('vertices','faces','origin','voxel_size','coords','attrs','voxel_shape','layout')}
        state={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        if isinstance(state['origin'],torch.Tensor): state['origin']=state['origin'].tolist()
        torch.save(state,raw)
        write_json(record,{**fingerprint,'model_revision':pin['revision'],'inference_seconds':time.time()-start,
                           'vertices':len(mesh.vertices),'faces':len(mesh.faces),'native_saved':True})
        del pipeline; torch.cuda.empty_cache()
    if not glb_path.exists():
        glb=o_voxel.postprocess.to_glb(vertices=mesh.vertices,faces=mesh.faces,attr_volume=mesh.attrs,
            coords=mesh.coords,attr_layout=mesh.layout,voxel_size=mesh.voxel_size,
            aabb=[[-.5,-.5,-.5],[.5,.5,.5]],decimation_target=cfg['decimation_target'],
            texture_size=cfg['texture_size'],remesh=True,remesh_band=1,remesh_project=0,verbose=True)
        glb.export(str(glb_path),extension_webp=True)
    # Keep official material render separate from the scene's Gaussian preview.
    if not (out/'trellis_asset_preview.png').exists():
        import cv2
        import imageio.v2 as imageio
        from trellis2.utils import render_utils
        from trellis2.renderers import EnvMap
        envmap=EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(str(repo/'assets/hdri/forest.exr'),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda'))
        video=render_utils.make_pbr_vis_frames(render_utils.render_video(mesh,envmap=envmap,num_frames=72))
        imageio.imwrite(out/'trellis_asset_preview.png',video[0])
        imageio.mimsave(out/'trellis_asset.mp4',video,fps=12)
    if cfg.get('method')=='pbr_multiview_gaussian':
        print('PBR GLB ready; Gaussian training belongs to M3 and placement to M4',flush=True)
        return
    from .mesh_to_gaussian import convert
    from .alignment import fit
    if not (out/'object_local.ply').exists():
        convert(glb_path,out/'object_local.ply',cfg,ctx.cfg['project']['seed'])
    if not (out/'object_world.ply').exists():
        from .scene import read_vertices
        from .m2_space import generation_frame
        fit(read_vertices(out/'object_local.ply'),generation_frame(ctx),
            read_json(ctx.stage('m2')/'anchor_camera.json'),ctx.stage('m2')/'object_mask.png',cfg,out)
    print('TRELLIS mesh and Gaussian adapter finished',flush=True)


if __name__=='__main__': run(Context(arguments().config))
