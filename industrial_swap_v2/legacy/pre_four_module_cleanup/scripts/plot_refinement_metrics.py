"""Plot measured training/validation quantities, never substitute perceptual claims."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..core.context import Context, read_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args(); ctx=Context(args.config)
    out=ctx.out/'validation'; out.mkdir(exist_ok=True)
    folder=ctx.stage('m4')/'background_optimization'
    data=read_json(folder/'optimization_log.json'); selected=read_json(folder/'summary.json')['selected_step']
    rows=[{'step':0,'summary':data['baseline']['summary']},*data['evaluations']]
    steps=[r['step'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for split,label in [('train','Training photos'),('validation','Validation photos')]:
        baseline=data['baseline']['summary'][split]['background_l1']
        axes[0].plot(steps,[(r['summary'][split]['background_l1']/baseline-1)*100 for r in rows],marker='.',label=label)
    axes[0].axhline(0,color='gray',lw=.8); axes[0].set_ylabel('Reliable background L1 change (%)')
    for split,label in [('train','Training photos'),('validation','Validation photos'),('anchor','Anchor')]:
        axes[1].plot(steps,[r['summary'][split]['visibility']*100 for r in rows],marker='.',label=label)
    axes[1].set_ylabel('Device visible contribution (%)')
    for ax in axes:
        ax.axvline(selected,color='black',ls='--',alpha=.5,label='Selected checkpoint')
        ax.set_xlabel('Stage A iteration'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'stage_a_metrics.png',dpi=180); plt.close(fig)
    path=ctx.stage('m4')/'training.jsonl'
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,key,title in zip(axes,['rgb_mse','mask_mse'],['Anchor RGB MSE','Anchor mask MSE']):
            ax.plot([r['step'] for r in rows],[r[key] for r in rows]); ax.set_xlabel('Stage B iteration')
            ax.set_ylabel(title); ax.grid(alpha=.2)
        fig.savefig(out/'stage_b_losses.png',dpi=180); plt.close(fig)


if __name__=='__main__': main()
