"""Batch adapter invoking the actual upstream WebUI Delete/Add implementations.

No rendering, guidance, training, generation or placement method is mocked.
Only Viser display handles are replaced with noninteractive controls.
"""
if __name__ == '__main__':
    import os,sys,json,argparse,random,ast
    from pathlib import Path
    from types import SimpleNamespace
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('stage',choices=['delete','add']);a=p.parse_args()
    ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';repo=BASE/'vendor/GaussianEditor';out=BASE/'gaussianeditor'/a.case
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
    os.environ['PATH']='/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    os.environ['TORCH_EXTENSIONS_DIR']=str(BASE/'cache/torch_extensions')
    os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
    os.environ['MAX_JOBS']='4'
    os.chdir(repo);sys.path.insert(0,str(repo))
    import numpy as np,torch
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    import webui,ui_utils
    from webui import WebUI,GaussianModel,CamScene,PipelineParams,LangSAMTextSegmentor
    from argparse import ArgumentParser
    task=json.loads((out/'task.json').read_text());geom=json.loads(Path(task['instance_geometry']).read_text())
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    class DisplayHandle:
     def remove(self):pass
    ui_utils.new_frustums=lambda *args,**kwargs:DisplayHandle()
    ui_utils.new_frustum_from_cam=lambda *args,**kwargs:(DisplayHandle(),DisplayHandle())
    w=WebUI.__new__(WebUI)
    # Read upstream widget defaults without creating a server or event callbacks.
    for node in ast.walk(ast.parse((repo/'webui.py').read_text())):
     if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
      for target in node.targets:
       if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
        for kw in node.value.keywords:
         if kw.arg=='initial_value':
          try:setattr(w,target.attr,SimpleNamespace(value=ast.literal_eval(kw.value),visible=False))
          except (ValueError,TypeError):pass
    for name in ['seg_scale_end_button','inpaint_end','depth_end']:
     setattr(w,name,SimpleNamespace(value=False,visible=False))
    w.server=SimpleNamespace(get_clients=lambda:{0:SimpleNamespace(camera=None)})
    w.use_sam=False;w.guidance=None;w.stop_training=False;w.ctn_inpaint=None;w.ip2p=None;w.points3d=[]
    w.seg_scale=True;w.seg_scale_end=True;w.inpaint_again=True;w.inpaint_end_flag=True;w.scale_depth=True;w.depth_end_flag=True
    w.seg_cam_num.value=20;w.mask_thres.value=.5
    w.pipe=PipelineParams(ArgumentParser());w.background_tensor=torch.zeros(3,device='cuda')
    w.colmap_dir=task['native_view_source'];scene=CamScene(w.colmap_dir,h=512,w=512)
    w.colmap_cameras=scene.cameras;w.cameras_extent=scene.cameras_extent
    w.gaussian=GaussianModel(sh_degree=0,anchor_weight_init_g0=1.,anchor_weight_init=.1,anchor_weight_multiplier=2)
    w.gaussian.load_ply(str(ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply') if a.stage=='delete' else str(out/'scene_deleted.ply'))
    w.gaussian.max_radii2D=torch.zeros(w.gaussian.get_xyz.shape[0],device='cuda')
    w.configure_optimizers()
    w.gaussian.update_anchor_term(w.anchor_weight_init_g0.value,w.anchor_weight_init.value,w.anchor_weight_multiplier.value)
    (out/'renders').mkdir(exist_ok=True)
    controls={k:v.value for k,v in vars(w).items() if isinstance(v,SimpleNamespace) and hasattr(v,'value')}
    (out/'records'/f'{a.stage}_controls.json').write_text(json.dumps(controls,indent=2))
    if a.stage=='delete':
     w.text_segmentor=LangSAMTextSegmentor().to('cuda')
     masks,selected=w.update_mask(task['A']);assert int(selected.sum())>0,'Empty native semantic tracing'
     np.save(out/'native_target_indices.npy',selected.nonzero().flatten().cpu().numpy())
     for i,cam in enumerate(w.colmap_cameras):
      pkg=w.render(cam);to_pil_image(pkg['semantic'][0].cpu()).save(out/'renders'/f'traced_{i:02}.png')
     del w.text_segmentor;torch.cuda.empty_cache()
     w.edit_text.value=task['background_text']
     w.delete(w.colmap_cameras,[DisplayHandle() for _ in w.colmap_cameras],[DisplayHandle() for _ in w.colmap_cameras])
     w.gaussian.anchor_postfix();w.gaussian.save_ply(str(out/'scene_deleted.ply'))
     for i,im in w.guidance.edit_frames.items():to_pil_image(im[0].permute(2,0,1).cpu()).save(out/'renders'/f'delete_target_{i:02}.png')
    else:
     # Select a registered target-facing camera; emulate native rectangular Add UI.
     center=np.array(geom['center']);half=np.array(geom['half_extents']);basis=np.array(geom['basis'])
     from itertools import product
     corners=center+np.array(list(product([-1,1],repeat=3)))*half @ basis.T
     choices=[]
     for cam in w.colmap_cameras:
      xyz=corners @ cam.R+cam.T
      if (xyz[:,2]<=0).any():continue
      focal=np.array([cam.image_width/(2*np.tan(cam.FoVx/2)),cam.image_height/(2*np.tan(cam.FoVy/2))])
      uv=xyz[:,:2]/xyz[:,2:]*focal+np.array([cam.image_width,cam.image_height])/2
      lo=uv.min(0);hi=uv.max(0);mid=(hi+lo)/2;span=(hi-lo)*1.04
      # Native Add needs a complete region, not the largest boundary-clipped box.
      margin=.04*np.array([cam.image_width,cam.image_height])
      if (mid-span/2<margin).any() or (mid+span/2>np.array([cam.image_width,cam.image_height])-margin).any():continue
      lo=(mid-span/2).astype(int);hi=(mid+span/2).astype(int)
      area=float(np.prod(hi-lo));choices.append((area,cam,lo,hi))
     assert choices, 'No fully framed target in the selected native Add cameras'
     _,cam,lo,hi=max(choices,key=lambda x:x[0])
     w.left_up.value=lo.tolist();w.right_down.value=hi.tolist();w.edit_text.value=task['B'];w.refine_text.value=task['gaussianeditor_refine_text'];w.inpaint_seed.value=42
     depth=float((center @ cam.R+cam.T)[2]);os.environ['GE_TARGET_CAMERA_DEPTH']=str(depth)
     record={'camera':cam.image_name,'R':cam.R.tolist(),'T':cam.T.tolist(),'FoVx':cam.FoVx,'FoVy':cam.FoVy,'width':cam.image_width,'height':cam.image_height,'add_bbox':[lo.tolist(),hi.tolist()],'target_camera_depth':depth,'add_text':task['B'],'refine_text':task['gaussianeditor_refine_text']}
     if (out/'add/inpainted.png').exists() and (out/'add/removed_bg.png').exists():
      assert json.loads((out/'records/add_inputs.json').read_text())==record, 'Cached Add inputs changed: archive this attempt before generating again'
      os.environ['GE_REUSE_ADD_ASSETS']='1'
     (out/'records/add_inputs.json').write_text(json.dumps(record,indent=2))
     w.add(cam);w.gaussian.anchor_postfix();w.gaussian.save_ply(str(out/'scene_final.ply'))
     for i,cam in enumerate(w.colmap_cameras):to_pil_image(w.render(cam)['comp_rgb'][0].permute(2,0,1).clamp(0,1).cpu()).save(out/'renders'/f'final_{i:02}.png')
    print(a.case,a.stage,'complete',flush=True)
