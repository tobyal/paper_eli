import numpy as np
from figure_common import *

def main():
    d=folder('F14_background_refinement');base=Path(CASES[0]['ours_source'])/'m4';rgb=d/'rgb_only_run/m4/background_optimization'
    full_summary=read(base/'summary.json');rgb_summary=read(rgb/'summary.json');baseline=read(base/'background_optimization/baseline.json')
    logs={'Full M4':read(base/'background_optimization/optimization_log.json')['evaluations'],
          'RGB only':read(rgb/'optimization_log.json')['evaluations']}
    fig=plt.figure(figsize=(9,9.2));gs=fig.add_gridspec(4,6,height_ratios=[.8,1.25,.95,1.],left=.095,right=.98,bottom=.13,top=.92,hspace=.52,wspace=.65);axes=[]
    fig.suptitle('Occlusion-Aware Local Background Refinement',y=.975,fontsize=14,fontweight='bold')
    for i,(name,label) in enumerate([('old_mask','Old visible mask'),('new_mask','New object mask'),('environment_mask','Environment mask')]):
        a=fig.add_subplot(gs[0,i*2:i*2+2]);axes.append(a);a.imshow(Image.open(d/'panels'/(name+'.png')),cmap='gray',vmin=0,vmax=255);a.axis('off');a.set_title('(a) '+label,fontsize=9)
    for i,(name,label) in enumerate([('step0','Step 0'),('rgb_only','RGB only - 2000 steps'),('full_candidate','Full M4 candidate - 2000 steps')]):
        a=fig.add_subplot(gs[1,i*2:i*2+2]);axes.append(a);a.imshow(Image.open(d/'panels'/(name+'_crop.png')));a.axis('off');a.set_title('(b) '+label,fontsize=9)
    rows=[]
    for i,(metric,label) in enumerate([('environment_l1','Environment RGB L1'),('trusted_visibility','Trusted object visibility')]):
        a=fig.add_subplot(gs[2,i*3:i*3+3]);axes.append(a)
        for j,(name,ev) in enumerate(logs.items()):
            x=[0]+[e['step'] for e in ev];y=[baseline['summary']['validation'][metric]]+[e['summary']['validation'][metric] for e in ev]
            a.plot(x,y,color=COLORS[j],label=name,lw=1.5)
        a.set(title='(c) Held-out '+label.lower(),xlabel='Iteration',ylabel=label);a.grid(alpha=.18);a.legend(frameon=False)
    dif=np.load(d/'background_differences.npz');env=dif['environment']
    for i,(name,label) in enumerate([('rgb_only','RGB only'),('full_candidate','Full M4 candidate'),('guarded_final','Guarded final (Step 0)')]):
        a=fig.add_subplot(gs[3,i*2:i*2+2]);axes.append(a);im=a.imshow(dif[name],cmap='magma',vmin=0,vmax=.05);a.axis('off');a.set_title('(d) '+label,fontsize=9)
        a.text(.5,-.10,f"Outside-mask MAE: {dif[name][env].mean():.5f}",transform=a.transAxes,ha='center',fontsize=8)
    cbax=fig.add_axes([.87,.075,.10,.012]);fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,.05]);cbax.set_title('RGB MAE',fontsize=7)
    for name,ev in logs.items():
        for e in ev:
            for split,s in e['summary'].items():rows.append(dict(record='evaluation',method=name,step=e['step'],split=split,guard_accepted=e['acceptable'],**s))
    for name,s,step,guard in [('Step 0',baseline['summary'],0,True),('Full candidate',full_summary['candidate'],2000,full_summary['candidate_guard_passed']),('Full guarded final',full_summary['guarded_final'],full_summary['selected_step'],True),('RGB guarded final',rgb_summary['final']['summary'],rgb_summary['selected_step'],True)]:
        for split,m in s.items():rows.append(dict(record='final_state',method=name,step=step,split=split,guard_accepted=guard,**m))
    fig.text(.075,.035,'Both candidates fail at least one per-view preservation guard. The guarded export retains Step 0.\nPlots show actual candidates separately; no foreground deletion, floor completion, or device-parameter updates.',fontsize=8)
    caption='Figure 14. Background-only optimization with a fixed Case00 device, pose, supervision masks, view sampling seed, learning rates, and 2,000-step budget. (a) The old/new visible masks and their union complement. (b) Same-camera crops of initial insertion, the new RGB-only control, and the existing full dual-mask candidate. RGB-only disables visibility and occlusion-specific guard/regularization losses while retaining environment RGB and parameter regularization. (c) Held-out-view mean environment L1 and trusted visibility measured every 50 steps, not smoothed sampled-view training losses. (d) Full-image absolute RGB differences from Step 0 outside the mask union, using a shared display range. The full candidate improves device visibility, but both controls fail at least one existing per-view preservation guard; guarded exports select Step 0. Candidate renders must not be described as guard-passed final results. Masks rely on reconstruction evidence and are not independent human annotations.'
    export(fig,d,'F14',caption,dict(render=read(d/'render_sources.json'),full_summary=source(base/'summary.json'),rgb_summary=source(rgb/'summary.json'),ablation=read(d/'rgb_only_run/ablation.json'),guarded_candidate_distinguished=True),rows,axes)

if __name__=='__main__':main()
