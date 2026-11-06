"""Render remaining base occlusion and actual unsupported-background contribution."""
import argparse
import numpy as np
import torch
from ..core.context import Context, gsle, read_json, write_json
from .validate_multiview_refinement import board


@torch.no_grad()
def inspect(ctx):
    c=gsle(ctx); out=ctx.out/'validation'; out.mkdir(exist_ok=True)
    folder=ctx.stage('m3')/'multiview_visibility'
    obj=c.GaussianData.load(ctx.stage('m3')/'object_world.ply')
    manifest=read_json(ctx.stage('m1')/'manifest.json')
    height=obj.xyz@torch.tensor(manifest['world_up'],device='cuda')
    base=height<torch.quantile(height,.2); base_colors=base[:,None].float().expand(-1,3).contiguous()
    camera=c.Camera.from_record(read_json(ctx.stage('m2')/'anchor_camera.json')).resize(1024,1024)
    standalone=c.raster(camera,obj,base_colors)[0]; denominator=standalone.sum().clamp_min(1)
    object_alpha=c.raster(camera,obj,torch.ones((len(obj.xyz),3),device='cuda'))[0]
    unsupported=np.load(folder/'unsupported_local_indices.npy'); supported=np.load(folder/'local_indices.npy')
    images=[]; labels=[]; results=[]
    for stage,path in [('M3',ctx.stage('m1')/'background_gaussians.ply'),('M4 A',ctx.stage('m4')/'background_optimization/background_optimized.ply')]:
        bg=c.GaussianData.load(path); joint=c.concatenate(bg,obj)
        visible=c.raster(camera,joint,torch.cat((torch.zeros((len(bg.xyz),3),device='cuda'),base_colors)))[0]
        groups=torch.zeros((len(bg.xyz)+len(obj.xyz),3),device='cuda')
        groups[torch.tensor(unsupported,device='cuda'),0]=1
        groups[torch.tensor(supported,device='cuda'),1]=1
        groups[len(bg.xyz):,2]=1
        actual=c.raster(camera,joint,groups)
        mask=object_alpha>.95
        images.extend([c.render_joint(camera,bg,obj,aux=False)['rgb'],actual])
        labels.extend([stage+' joint scene',stage+' contributions: red=frozen local, green=optimized, blue=device'])
        results.append({'stage':stage,'base_visibility':float(visible.sum()/denominator),
            'unsupported_contribution_in_opaque_object':float((actual[0]*mask).sum()/mask.sum().clamp_min(1)),
            'supported_contribution_in_opaque_object':float((actual[1]*mask).sum()/mask.sum().clamp_min(1)),
            'device_contribution_in_opaque_object':float((actual[2]*mask).sum()/mask.sum().clamp_min(1))})
        del bg,joint; torch.cuda.empty_cache()
    board(images,labels).save(out/'background_support_contributions.png')
    write_json(out/'background_support_report.json',{'anchor_resolution':1024,'results':results,
        'base_definition':'Lowest 20% of Gaussian centers along M1 world_up',
        'interpretation':'Diagnostic rendered contributions include possible residual background transmission; they are not ground-truth segmentation or complete counterfactual attribution.'})
    print(results)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args(); inspect(Context(args.config))
