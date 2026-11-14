"""Shared intended-workstation metrics from final saved PLYs, never matched-view screenshots."""
import sys,gc,os
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.ndimage import binary_dilation
from figure_common import ROOT,BASE,OUT,CASES,read,save,source,csvout,m4folder
sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,gsle
from industrial_swap_v2.core.scene import read_vertices
from industrial_swap_v2.core.m2_space import registration_space

def main():
    import lpips
    os.environ['TORCH_HOME']=str(Path.home()/'.cache/torch')
    net=lpips.LPIPS(net='alex').cuda().eval()
    out=OUT/'tables';out.mkdir(exist_ok=True);(out/'metric_renders').mkdir(exist_ok=True)
    c=gsle(Context(Path(CASES[0]['ours_source'])/'config.yaml'))
    source_path=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    original=c.GaussianData.load(source_path);metrics=[];provenance=[]
    sam=read(OUT/'F08_localization/localization_metrics.json')
    for row in CASES:
        case=row['case'];ctx=Context(Path(row['ours_source'])/'config.yaml');space=read(m4folder(row)/'alignment/report.json')['space'];f=space['frame']
        ids=np.load(Path(row['target_record']).parent/'target_indices.npy')
        old=original.subset(ids);obj=c.GaussianData.load(Path(row['ours_scene']).parent/'object_world.ply')
        cameras=read(ROOT/f'baseline_comparison/comparison_updated_v8/{case}/camera_params.json')['methods']['ours']
        cameras=cameras.get('views',cameras)
        views=[]
        with torch.no_grad():
            for name in ['main','aux1','aux2']:
                cam=c.Camera.from_record(cameras[name]['final_camera']).resize(512,512)
                rgb=c.render(cam,original,aux=False)['rgb'];a=c.raster(cam,old,torch.ones_like(old.xyz))[0];b=c.raster(cam,obj,torch.ones_like(obj.xyz))[0]
                excluded=binary_dilation(((a>.05)|(b>.05)).cpu().numpy(),iterations=3);env=torch.tensor(~excluded,device='cuda')
                views.append((name,cam,rgb,env))
                Image.fromarray((~excluded).astype(np.uint8)*255).save(out/'metric_renders'/f'{case}_{name}_env.png')
        del obj,old;gc.collect();torch.cuda.empty_cache()
        for method in ['gaussianeditor','dge','gsle','ours']:
            p=Path(row['ours_scene']) if method=='ours' else ROOT/f'baseline_comparison/{method}_native/{case}'/('composition/scene.ply' if method=='gsle' else 'final_scene.ply')
            op=Path(row['ours_scene']).parent/'object_world.ply' if method=='ours' else (ROOT/f'baseline_comparison/{method}_native/{case}'/('coarse/final_object_world.ply' if method=='gsle' else 'add/object_world.ply'))
            data=c.GaussianData.load(p);vals=[]
            with torch.no_grad():
                for name,cam,reference,env in views:
                    rgb=c.render(cam,data,aux=False)['rgb'];mask=env.float()[None]
                    # Both excluded regions are the same neutral gray; LPIPS sees no edit content.
                    aa=(reference*2-1)*mask;bb=(rgb*2-1)*mask
                    score=float(net(aa[None],bb[None]));mae=float((rgb-reference).abs()[:,env].mean())
                    vals.append(dict(case=case,method=method,view=name,masked_background_lpips=score,background_rgb_mae=mae,environment_fraction=float(env.float().mean())))
                    c.save_image(out/'metric_renders'/f'{case}_{method}_{name}.png',rgb)
            overflow='NA'
            if op.exists():
                v=read_vertices(op);xyz=np.stack([v[k] for k in ['x','y','z']],1);local=(xyz-np.asarray(f['center']))@np.asarray(f['axes'])
                outside=((local<np.asarray(f['local_bbox_low']))|(local>np.asarray(f['local_bbox_high']))).any(1)
                opacity=1/(1+np.exp(-np.clip(v['opacity'],-50,50)));overflow=float(np.sum(outside*opacity)/opacity.sum())
            agreement=next(r for r in sam if r['case']==case and r['method']==method)['reference_sam_iou']
            metrics.append(dict(case=case,method='GSLE (coarse)' if method=='gsle' else method,Localization_IoU='NA',Reference_SAM_IoU=agreement,
                Replacement_Success_Rate='NA',Text_Attribute_Accuracy='NA',Workspace_Overflow_OpacityWeightedCenters=overflow,
                Background_LPIPS=float(np.mean([x['masked_background_lpips'] for x in vals])),Background_RGB_MAE=float(np.mean([x['background_rgb_mae'] for x in vals])),views=3,
                Guard_Status='candidate_failed_guard' if method=='ours' else 'not_applicable',
                note='SAM metric is in-sample. Overflow tests intended Ours workstation and Gaussian centers, not occupied volume. Shared-camera LPIPS excludes common old/new Ours silhouettes.'))
            provenance.append(dict(case=case,method=method,scene=source(p),object=source(op) if op.exists() else None,space=space,views=vals,cameras={n:cam.record() for n,cam,_,_ in views}))
            save(out/'overall_metrics_partial.json',metrics);save(out/'overall_sources_partial.json',provenance)
            print(case,method,metrics[-1]['Background_LPIPS'],flush=True)
            del data;gc.collect();torch.cuda.empty_cache()
        del views;gc.collect();torch.cuda.empty_cache()
    csvout(out/'T01_overall.csv',metrics);save(out/'overall_metrics.json',metrics)
    save(out/'overall_sources.json',dict(source_scene=source(source_path),cases=provenance,
        lpips_alexnet_weights=source(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'),
        policy='3 identical scene cameras per case for quantitative evaluation, regardless of qualitative viewpoint matching. RGB normalized to [-1,1]; excluded pixels set to 0 in both images. No ground-truth replacement image exists. Low background LPIPS alone does not indicate successful replacement.',
        independent_annotations_available=False,original_scenes_modified=False))

if __name__=='__main__':main()
