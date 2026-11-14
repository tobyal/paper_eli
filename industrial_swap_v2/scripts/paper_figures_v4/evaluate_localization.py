"""Same-camera, actual Gaussian contribution comparison to input SAM references."""
import os,sys,gc
from pathlib import Path
import numpy as np,torch
from PIL import Image
from figure_common import ROOT,BASE,OUT,CASES,folder,read,save,source,csvout
sys.path.insert(0,str(ROOT))
from industrial_swap_v2.core.context import Context,gsle

def main():
    d=folder('F08_localization');ctx=Context(Path(CASES[0]['ours_source'])/'config.yaml');c=gsle(ctx)
    original=ROOT/'outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply'
    scene=c.GaussianData.load(original);rows=[];provenance=[]
    with torch.no_grad():
        for row in CASES:
            case=row['case'];dest=d/'cases'/case;dest.mkdir(parents=True,exist_ok=True)
            camera=c.Camera.from_record(row['reference_camera']);save(dest/'camera.json',camera.record())
            rgb=c.render(camera,scene,aux=False)['rgb'];c.save_image(dest/'original.png',rgb)
            reference=np.load(row['reference_sam_mask'])
            if reference.ndim==3:reference=reference.squeeze()
            reference=np.asarray(Image.fromarray((reference.astype(bool)*255).astype(np.uint8)).resize((camera.width,camera.height),Image.Resampling.NEAREST))>0
            Image.fromarray(reference.astype(np.uint8)*255).save(dest/'reference_mask.png')
            for method in ('gaussianeditor','dge','gsle','ours'):
                p=Path(row['target_record']).parent/'target_indices.npy' if method=='ours' else ROOT/'baseline_comparison'/(method+'_native')/case/('localization/selected_indices.npy' if method=='gaussianeditor' else 'target_indices.npy')
                ids=np.load(p);flags=torch.zeros_like(scene.xyz);flags[torch.as_tensor(ids,device='cuda')]=1
                contribution=c.raster(camera,scene,flags)[0].cpu().numpy();mask=contribution>.05
                Image.fromarray(mask.astype(np.uint8)*255).save(dest/(method+'_mask.png'))
                # Original splat colors, semi-transparent red tint on exact selection.
                colors=scene.colors(camera.center);colors[ids]=.5*colors[ids]+.5*colors.new_tensor([1.,0.,0.])
                c.save_image(dest/(method+'_highlight.png'),c.raster(camera,scene,colors))
                n=(mask&reference).sum();record=dict(case=case,method=method,selected_gaussians=len(ids),reference_pixels=int(reference.sum()),visible_pixels=int(mask.sum()),
                    reference_sam_iou=float(n/max((mask|reference).sum(),1)),reference_sam_recall=float(n/max(reference.sum(),1)),reference_sam_precision=float(n/max(mask.sum(),1)),
                    supervision='input SAM reference, not independent ground truth',visible_contribution_threshold=.05,reference_camera=camera.name)
                rows.append(record);provenance.append(dict(case=case,method=method,indices=source(p),sam=source(row['reference_sam_mask']),camera=camera.record()))
                print(case,method,record['reference_sam_iou'],flush=True)
            del rgb,flags,colors;gc.collect();torch.cuda.empty_cache()
    csvout(d/'localization_metrics.csv',rows);save(d/'localization_metrics.json',rows)
    save(d/'evaluation_sources.json',dict(source_scene=source(original),methods=provenance,
        text_only='Existing GSLE-native: three-scale frozen LangSplat, text-vs-negative relevance, geometry-only connected-component selection. No reference photo or SAM used.',
        limitation='Reference SAM participated in photo-assisted localization. This is an in-sample diagnostic; no independent-instance accuracy claim.'))

if __name__=='__main__':main()
