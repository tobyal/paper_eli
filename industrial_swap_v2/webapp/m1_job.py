"""M1-only subprocess coordinator. No generation stages are imported."""
import argparse
import os
from pathlib import Path
import subprocess
from ..core.context import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    job=read_json(a.job);root=Path(job['root']);folder=a.job.parent
    if job['stages']!=['m1']:raise ValueError('This executor is restricted to M1')
    env=os.environ.copy();env.update(SWAP_DISABLE_QWEN='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([str(root/'envs/gsle_langsplat/bin/python'),'-m','industrial_swap_v2.scripts.m1_language',
        '--root',str(root),'--output',str(folder/'semantic'),'--selection',job['selection'],'--text',job['text']],cwd=root,env=env,check=True)
    subprocess.run([str(root/'envs/gsle/bin/python'),'-m','industrial_swap_v2.scripts.m1_assisted_localize','--job',str(a.job)],cwd=root,env=env,check=True)


if __name__=='__main__':main()
