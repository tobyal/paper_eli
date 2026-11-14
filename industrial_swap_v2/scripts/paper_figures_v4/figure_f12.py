"""F12 from actual initial and selected PBR checkpoint evaluations."""
import numpy as np
from figure_common import *

def main():
    d=folder('F12_m3_training');rows=[];sources=[];summaries=[]
    for row in CASES:
        m3=Path(row['ours_source'])/'m3';s=read(m3/'pbr_training_summary.json');summaries.append(s)
        sources.append(dict(case=row['case'],summary=source(m3/'pbr_training_summary.json'),
            evaluations=source(m3/'evaluations.json'),training=source(m3/'training.jsonl'),
            selected_checkpoint_step=s['best_step'],trained_steps=s['iterations']))
        for stage,key,step in [('Initial','baseline',0),('PBR optimized','best',s['best_step'])]:
            for split in ['train','validation']:
                rows.append(dict(record='summary',case=row['case'],stage=stage,step=step,split=split,**s[key][split]))
        for ev in read(m3/'evaluations.json'):
            for split in ['train','validation']:
                rows.append(dict(record='evaluation',case=row['case'],stage='PBR optimized',step=ev['step'],split=split,**ev[split]))
    m3=Path(CASES[0]['ours_source'])/'m3';training=lines(m3/'training.jsonl');evals=read(m3/'evaluations.json')
    for x in training:rows.append(dict(record='per_iteration',case='case00',stage='PBR optimized',split='sampled_training_view',**x))
    fig,ax=plt.subplots(2,2,figsize=(8.5,6.4));fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.87,wspace=.30,hspace=.47)
    fig.suptitle('Multi-view Gaussian Asset Reconstruction',y=.97,fontsize=14,fontweight='bold')
    for a,loss,label,ekey,title in [(ax[0,0],'rgb_l1','RGB L1','rgb_l1','(a) RGB supervision - Case00'),
        (ax[0,1],'depth_relative_l1','Depth L1 / object diagonal','depth_mae_relative','(b) Depth supervision - Case00')]:
        vals=np.array([x[loss] for x in training]);steps=np.array([x['step'] for x in training])
        a.plot(steps,vals,color=COLORS[0],alpha=.16,lw=.45)
        a.plot(steps[49:],np.convolve(vals,np.ones(50)/50,mode='valid'),color=COLORS[0],lw=1.5,label='Train (50-step mean)')
        a.plot([0]+[x['step'] for x in evals],[summaries[0]['baseline']['validation'][ekey]]+[x['validation'][ekey] for x in evals],color=COLORS[1],marker='o',ms=3,lw=1.3,label='Held-out views')
        a.axvline(summaries[0]['best_step'],color='#777777',ls=':',lw=1,label='Selected checkpoint')
        a.set(title=title,xlabel='Training iteration',ylabel=label);a.grid(alpha=.18);a.legend(frameon=False)
    x=np.arange(5);width=.35;labels=['00\nRobot','02\nConveyor','04\nChiller','07\nStorage','10\nTool cabinet']
    for a,key,factor,ylabel,title in [(ax[1,0],'foreground_psnr',1,'Foreground PSNR (dB)','(c) Held-out RGB reconstruction'),
        (ax[1,1],'depth_mae_relative',100,'Depth MAE / object diagonal (%)','(d) Held-out depth reconstruction')]:
        for offset,k,label,color in [(-.5,'baseline','Initial',COLORS[4]),(.5,'best','PBR optimized',COLORS[0])]:
            vals=[s[k]['validation'][key]*factor for s in summaries]
            a.bar(x+offset*width,vals,width,label=label,color=color)
        a.set_xticks(x,labels);a.set(ylabel=ylabel,title=title);a.grid(axis='y',alpha=.18);a.set_axisbelow(True);a.legend(frameon=False)
    fig.text(.10,.025,'30 training + 6 held-out PBR views per asset. Optimized = saved validation-selected checkpoint; no new training.',fontsize=8)
    caption='Figure 12. Multi-view Gaussian asset reconstruction using existing PBR RGB-D training. (a,b) Case00 sampled training losses (faint raw values; 50-step moving means) and held-out-view evaluations. (c,d) Initial versus saved validation-selected Gaussian checkpoints for five assets. Depth error is normalized by each local asset diagonal for comparability. All runs used 30 training views, six held-out views, and 4,000 training iterations. These metrics evaluate fidelity to the fixed GLB/PBR rendering targets, not independent real-world geometry or relighting accuracy. The selected checkpoint can precede iteration 4,000; exact steps and absolute depth MAE are included in the CSV. RGB-only training was not run and is not implied.'
    export(fig,d,'F12',caption,dict(cases=sources,no_new_training=True,metric_reference='Existing GLB rendered under fixed HDRI, not real captured photographs'),rows,ax.ravel())

if __name__=='__main__':main()
