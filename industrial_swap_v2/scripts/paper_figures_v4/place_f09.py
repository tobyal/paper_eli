"""Fixed Case00 support/center/volume; yaw-scale fitting and unchanged-background composition."""
import sys,copy,gc,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
from industrial_swap_v2.core import gaussian_registration as reg
from industrial_swap_v2.core.alignment import transform_vertices
from industrial_swap_v2.core.scene import read_vertices,save_vertices,compose
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F09_text_control'

@torch.no_grad()
def fixed_center_volume_projection(model,p):
    """Enforce the volume gate by scale alone when horizontal motion is locked."""
    _,_,positions,sigma=model.spatial(p);scale=p[1].exp()
    lo=(torch.quantile(positions-sigma,.002,dim=0)-model.support)/scale
    hi=(torch.quantile(positions+sigma,.998,dim=0)-model.support)/scale
    tolerance=model.cfg['max_volume_fraction']*.995
    limits=[scale]
    for axis in range(3):
        if lo[axis]<-1e-8:limits.append((model.low[axis]-tolerance*model.extent[axis]-model.support[axis])/lo[axis])
        if hi[axis]>1e-8:limits.append((model.high[axis]+tolerance*model.extent[axis]-model.support[axis])/hi[axis])
    limit=torch.stack(limits).min()
    if limit<=0:raise ValueError('Fixed support center is outside the admissible workspace')
    p[1]=limit.log();p[2:]=0

def main():
    row=read_json(ROOT/'baseline_comparison/prompts.json')['cases'][0];old=Path(row['ours_source'])
    original_pose=read_json(old/'m4/transform.json');space=read_json(old/'m4/alignment/report.json')['space']
    frame=space['frame'];axes=np.asarray(frame['axes']);center=np.asarray(frame['center'])
    support=np.asarray(original_pose['support_center_world']);space['support_center_local']=((support-center)@axes).tolist()
    space['bottom_local_y']=space['support_center_local'][1]
    space['source']='Case00 saved robot registration volume and exact accepted installation support center'
    write_json(OUT/'fixed_installation_space.json',dict(space=space,support_world=support.tolist(),original_transform=original_pose,
        policy='All four variants use the same original M1 retained background, world installation center, support height, volume and camera. Only scale and yaw change.'))
    contexts={'A_robot':Context(old/'config.yaml')}
    for key in (sys.argv[1:] or ['B_scara','C_cnc','D_conveyor']):
        ctx=Context(OUT/'variants'/key/'config.yaml');contexts[key]=ctx
        if not (ctx.stage('m3')/'pbr_training_summary.json').exists():raise FileNotFoundError(key+' M3 not yet complete')
        if (ctx.stage('m4')/'object_world.ply').exists():
            saved=read_json(ctx.stage('m4')/'transform.json')
            if saved['max_volume_overflow_fraction']<=reg.DEFAULTS['max_volume_fraction']:continue
            # Resume a candidate whose earlier generic projection required disallowed XY motion.
            c=gsle(ctx);cfg={**reg.DEFAULTS,**ctx.cfg['m4']['integration']['registration']}
            source=ctx.stage('m3')/'object_refined_local.ply';obj=c.GaussianData.load(source);model=reg.Pose(c,obj,space,cfg)
            p=model.tensor(saved['parameters_radians_logscale_dx_dz']);fixed_center_volume_projection(model,p)
            camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'));ref=reg.reference(ctx,camera,640,'cuda')
            with torch.no_grad():_,metrics,pred=reg.evaluate(model,p,ref)
            pose=reg.pose_report(model,p);write_json(ctx.stage('m4')/'alignment/before_fixed_center_projection.json',saved)
            save_vertices(ctx.stage('m4')/'object_world.ply',transform_vertices(read_vertices(source),pose['scale'],np.asarray(pose['rotation']),np.asarray(pose['translation'])))
            write_json(ctx.stage('m4')/'transform.json',{**saved,**pose,'metrics':reg.numbers(metrics),'fixed_center_scale_projection':True})
            del model,obj;gc.collect();torch.cuda.empty_cache();continue
        c=gsle(ctx);reg.registration_space=lambda ignored:copy.deepcopy(space)
        try:
            reg.fit_rasterized(ctx)
        except ValueError as e:
            if 'Registration failed silhouette/space checks' not in str(e):raise
            # Preserve the real constrained candidate and failed quality gate; no fabricated fit success.
            search=read_json(ctx.stage('m4')/'alignment/search.json');cfg={**reg.DEFAULTS,**ctx.cfg['m4']['integration']['registration']}
            obj=c.GaussianData.load(ctx.stage('m3')/'object_refined_local.ply');model=reg.Pose(c,obj,space,cfg)
            p=model.tensor(search['selected']['parameters']);fixed_center_volume_projection(model,p)
            camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json'));ref=reg.reference(ctx,camera,640,'cuda')
            with torch.no_grad():_,metrics,pred=reg.evaluate(model,p,ref)
            pose=reg.pose_report(model,p);source=ctx.stage('m3')/'object_refined_local.ply'
            save_vertices(ctx.stage('m4')/'object_world.ply',transform_vertices(read_vertices(source),pose['scale'],np.asarray(pose['rotation']),np.asarray(pose['translation'])))
            write_json(ctx.stage('m4')/'transform.json',{**pose,'metrics':reg.numbers(metrics),'quality_gate_passed':False,'warning':str(e),'source_local_sha256':sha256(source),'fixed_installation':True})
            reg.save_render(ctx.stage('m4')/'alignment','constrained_candidate',pred,ref,c)
            del model,obj;gc.collect();torch.cuda.empty_cache()
        print(key,'PLACED',flush=True)
    if sys.argv[1:]:return
    c=gsle(contexts['A_robot']);camera_record=read_json(ROOT/'industrial_swap_v2/outputs/paper_figures_v3/F05_case00/sources.json')['main_camera']
    main_camera=c.Camera.from_record(camera_record);anchor=c.Camera.from_record(read_json(old/'m2/anchor_camera.json'))
    background=old/'m1/background_gaussians.ply';bg_hash=sha256(background);results=[];bounds=[]
    for key,ctx in contexts.items():
        dest=OUT/'variants'/key;dest.mkdir(parents=True,exist_ok=True)
        obj_path=old/'m4/object_world.ply' if key=='A_robot' else ctx.stage('m4')/'object_world.ply'
        obj_hash=sha256(obj_path);scene_path=dest/'scene_final.ply'
        if not scene_path.exists():compose(background,obj_path,scene_path,dest/'composition.json')
        obj=c.GaussianData.load(obj_path);scene=c.GaussianData.load(scene_path)
        with torch.no_grad():
            for label,cam in [('main',main_camera),('anchor',anchor)]:
                c.save_image(dest/(label+'_scene.png'),c.render(cam,scene,aux=False)['rgb'])
                c.save_image(dest/(label+'_object.png'),c.render(cam,obj,aux=False,background=torch.ones(3,device='cuda'))['rgb'])
                alpha=c.raster(cam,obj,torch.ones_like(obj.xyz))[0].cpu().numpy();Image.fromarray((alpha.clip(0,1)*255).astype(np.uint8)).save(dest/(label+'_alpha.png'))
                if label=='main':
                    y,x=np.nonzero(alpha>.05);bounds.append([x.min(),y.min(),x.max(),y.max()])
        transform=original_pose if key=='A_robot' else read_json(ctx.stage('m4')/'transform.json')
        assert np.allclose(transform['support_center_world'],support,atol=2e-6)
        assert sha256(obj_path)==obj_hash
        results.append(dict(variant=key,object_path=str(obj_path),object_sha256=obj_hash,scene_path=str(scene_path),scene_sha256=sha256(scene_path),
            transform=transform,background_sha256=bg_hash,gaussians=len(obj.xyz),M4_background_steps=0))
        write_json(dest/'camera_params.json',dict(main=main_camera.record(),anchor=anchor.record()))
        del scene,obj;gc.collect();torch.cuda.empty_cache()
    # Same crop for all variants: no per-device scale normalization.
    b=np.asarray(bounds);lo=b[:,:2].min(0);hi=b[:,2:].max(0);mid=(lo+hi)/2;size=int(max(hi-lo)*1.2)
    crop=[max(0,int(mid[0]-size/2)),max(0,int(mid[1]-size/2)),min(main_camera.width,int(mid[0]+size/2)),min(main_camera.height,int(mid[1]+size/2))]
    for r in results:Image.open(OUT/'variants'/r['variant']/'main_scene.png').crop(crop).save(OUT/'variants'/r['variant']/'detail.png')
    assert sha256(background)==bg_hash
    write_json(OUT/'placement_results.json',dict(variants=results,main_camera=main_camera.record(),common_crop=crop,shared_background=str(background),
        original_case00_asset_unchanged=True,original_case00_final_scene_not_overwritten=True,
        note='A reuses the original robot object/pose and is recomposed with the same M1 background as B/C/D for a controlled text experiment. It is not the M4-optimized scene shown in F05/F10.'))
    print('F09 PLACEMENT COMPLETE',flush=True)

if __name__=='__main__':main()
