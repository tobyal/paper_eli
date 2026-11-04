"""Diagnostic only: unchanged official GE guidance with equivalent prompt phrasings."""
import json
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
root=Path(os.environ['ROOT'])
parser=argparse.ArgumentParser()
parser.add_argument('--image',type=Path)
parser.add_argument('--output',default='prompt_diagnostic')
args=parser.parse_args()
os.chdir(root/'repos/GaussianEditor')
sys.path.insert(0,str(root/'repos/GaussianEditor'))
import threestudio
models=json.loads((root/'records/core_model_downloads.json').read_text())
guidance=threestudio.find('stable-diffusion-instructpix2pix-guidance')({
    'ip2p_name_or_path':models['timbrooks/instruct-pix2pix']['path'],
    'ddim_scheduler_name_or_path':models['CompVis/stable-diffusion-v1-4']['path'],
    'guidance_scale':7.5,'condition_scale':1.5,'min_step_percent':.02,'max_step_percent':.98})
path=root/'outputs/gaussianeditor/edit-n2n/industrial_red_seed0/save/it500-130.png'
if args.image: path=args.image if args.image.is_absolute() else root/args.image
image=Image.open(path).convert('RGB').crop((0,0,512,512))
rgb=torch.from_numpy(np.array(image).astype(np.float32)/255)[None].cuda()
prompts=[
 json.loads((root/'records/experiment.json').read_text())['instruction_en'],
 'Paint the white machine base red, keeping all other objects unchanged.',
 'Paint the white base of the selected industrial machine red, keeping all other objects unchanged.',
 'Make the machine base red.'
]
destination=root/'outputs'/args.output
destination.mkdir(exist_ok=True)
image.save(destination/'source.png')
records=[]
for index,prompt in enumerate(prompts):
    processor=threestudio.find('stable-diffusion-prompt-processor')({'pretrained_model_name_or_path':models['stable-diffusion-v1-5/stable-diffusion-v1-5']['path'],'prompt':prompt,'spawn':False})
    utils=processor()
    for seed in [0,1]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(): result=guidance(rgb,rgb,utils)['edit_images'][0].clamp(0,1)
        filename=f'prompt_{index}_seed_{seed}.png'
        Image.fromarray((result.cpu().numpy()*255).astype(np.uint8)).save(destination/filename)
        records.append({'prompt':prompt,'seed':seed,'file':filename,'role':'2D diagnostic, not a 3D method result'})
    del processor
(destination/'manifest.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2),flush=True)
