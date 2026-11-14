"""Actual Gaussian pose ablation and sampled objective surface; no asset training."""
import gc,math,sys,time
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,gsle,read_json,write_json,sha256
from industrial_swap_v2.core import gaussian_registration as reg
from industrial_swap_v2.core.m2_space import registration_space
DEST=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F13_spatial_alignment'

def main():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'panels').mkdir(exist_ok=True)
    rows=read_json(ROOT/'baseline_comparison/prompts.json')['cases'][:2]
    torch.cuda.set_per_process_memory_fraction(.35);results=[]
    for row in rows:
        print('ALIGN_ABLATION',row['case'],flush=True);t=time.time()
        base=Path(row['ours_source']);ctx=Context(base/'config.yaml');c=gsle(ctx)
        cfg={**reg.DEFAULTS,**ctx.cfg['m4']['integration'].get('registration',{})}
        objpath=base/'m3/object_refined_local.ply';original_hash=sha256(objpath)
        obj=c.GaussianData.load(objpath);space=registration_space(ctx);model=reg.Pose(c,obj,space,cfg)
        camera=c.Camera.from_record(read_json(base/'m2/anchor_camera.json'))
        ref=reg.reference(ctx,camera,320,'cuda');fine=reg.reference(ctx,camera,640,'cuda')
        selected=read_json(base/'m4/transform.json')['parameters_radians_logscale_dx_dz']
        poses={'A_bbox':model.tensor([0,math.log(model.base_scale),0,0]),'C_full':model.tensor(selected)}
        # Exactly the recorded official initialization pool, rescored only by silhouette.
        seeds=read_json(base/'m4/alignment/search.json')['coarse_search'];pool=[]
        with torch.no_grad():
            for seed in seeds:
                p=model.tensor(seed['parameters']);m=reg.image_metrics(model.render(p,ref['camera']),ref)
                pool.append((float(m['soft_iou']),p))
        pool.sort(key=lambda x:-x[0]);starts=[]
        for score,p in pool:
            if all(abs((float(p[0]-q[0])+math.pi)%(2*math.pi)-math.pi)>=math.radians(30) for q in starts):starts.append(p)
            if len(starts)==6:break
        candidates=[]
        for n,initial in enumerate(starts):
            p=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([p],lr=cfg['learning_rate']);best=(1e9,p.detach().clone())
            for step in range(120):
                optimizer.zero_grad();m=reg.image_metrics(model.render(p,ref['camera']),ref);loss=1-m['soft_iou']
                if float(loss.detach())<best[0]:best=(float(loss.detach()),p.detach().clone())
                loss.backward();torch.nn.utils.clip_grad_norm_([p],1.);optimizer.step();model.constrain(p)
            with torch.no_grad():score=float(reg.image_metrics(model.render(best[1],fine['camera']),fine)['soft_iou'])
            candidates.append((score,best[1]));print(row['case'],'silhouette candidate',n,score,flush=True)
        candidates.sort(key=lambda x:-x[0]);finalists=[]
        for _,initial in candidates[:2]:
            p=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([p],lr=cfg['learning_rate']/3);best=(1e9,p.detach().clone())
            for step in range(180):
                optimizer.zero_grad();m=reg.image_metrics(model.render(p,fine['camera']),fine);loss=1-m['soft_iou']
                if float(loss.detach())<best[0]:best=(float(loss.detach()),p.detach().clone())
                loss.backward();torch.nn.utils.clip_grad_norm_([p],1.);optimizer.step();model.constrain(p)
            finalists.append(best)
        poses['B_silhouette']=min(finalists,key=lambda x:x[0])[1]
        bg=c.GaussianData.load(base/'m1/background_gaussians.ply');cc=camera.resize(1024,1024)
        reference_mask=np.asarray(Image.open(base/'m2/object_mask.png').resize((1024,1024),Image.Resampling.NEAREST))>127
        contours,_=cv2.findContours(reference_mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        footprint_diag=float(torch.linalg.norm(model.extent[[0,2]]))
        with torch.no_grad():
            for method in ['A_bbox','B_silhouette','C_full']:
                p=poses[method];_,metrics,pred=reg.evaluate(model,p,fine);pr=reg.pose_report(model,p)
                _,_,positions,sigma=model.spatial(p)
                outside=((positions-sigma<model.low)|(positions+sigma>model.high)).any(1)
                support_fraction=float((outside*model.probe_weight).sum()/model.probe_weight.sum())
                offset=model.extent[[0,2]]*p[2:]
                results.append(dict(case=row['case'],method=method,**reg.numbers(metrics),
                    **pr,support_outside_fraction=support_fraction,
                    center_offset_fraction=float(torch.linalg.norm(offset))/footprint_diag))
                world=model.data(p);scale,rot,shift,_,_=model.placement(p)
                localcam=(model.tensor(cc.center)-shift)@rot/scale
                joint=c.concatenate(bg,world);colors=torch.cat((bg.colors(cc.center),obj.colors(localcam)))
                render=c.raster(cc,joint,colors);path=DEST/'panels'/f'{row["case"]}_{method}.png';c.save_image(path,render)
                arr=np.asarray(Image.open(path)).copy();cv2.drawContours(arr,contours,-1,(230,55,65),2)
                Image.fromarray(arr).save(DEST/'panels'/f'{row["case"]}_{method}_contour.png')
                del joint,colors,world
        if row['case']=='case02':
            # Actual objective slices at fixed selected horizontal translation.
            p=poses['C_full'];surface=[]
            with torch.no_grad():
                small=reg.reference(ctx,camera,160,'cuda')
                for yaw in np.linspace(float(p[0])-math.pi/2,float(p[0])+math.pi/2,25):
                    for factor in np.linspace(.65,1.35,21):
                        q=p.clone();q[0]=yaw;q[1]=p[1]+math.log(factor)
                        _,m,_=reg.evaluate(model,q,small)
                        surface.append(dict(yaw_degrees=math.degrees(yaw),scale=float(q[1].exp()),scale_factor=factor,**reg.numbers(m)))
            write_json(DEST/'objective_surface.json',dict(case=row['case'],resolution=160,fixed_translation=p[2:].tolist(),values=surface,selected=reg.pose_report(model,p)))
            old=c.GaussianData.load(base/'m1/target_gaussians.ply');old_local=(old.xyz-model.center)@model.axes
            _,_,new_local,_=model.spatial(p)
            np.savez_compressed(DEST/'workspace_points.npz',old=old_local.cpu().numpy()[::10],new=new_local.cpu().numpy(),low=model.low.cpu().numpy(),high=model.high.cpu().numpy())
        assert sha256(objpath)==original_hash
        write_json(DEST/(row['case']+'_provenance.json'),dict(source_run=str(base),source_asset=str(objpath.resolve()),source_asset_sha256=original_hash,seconds=time.time()-t,
            silhouette_only='Soft IoU only; same original seed pool and bounded yaw/logscale/horizontal parameterization; support plane remains fixed for all methods.',
            full_method='Existing saved registration; not reoptimized',background_optimization=False))
        del model,obj,bg;gc.collect();torch.cuda.empty_cache()
    write_json(DEST/'ablation_results.json',results);print('ALIGNMENT ABLATION COMPLETE',flush=True)

if __name__=='__main__':main()
