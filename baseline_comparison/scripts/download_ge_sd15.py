"""Complete pinned SD1.5 cache required by native ControlNet inpainting."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
repo='stable-diffusion-v1-5/stable-diffusion-v1-5'
info=json.loads((ROOT/'records/core_model_downloads.json').read_text())[repo]
revision=Path(info['path']).name
path=snapshot_download(repo,revision=revision,allow_patterns=['model_index.json','scheduler/*','tokenizer/*','text_encoder/config.json','text_encoder/pytorch_model.bin','unet/config.json','unet/diffusion_pytorch_model.bin','vae/config.json','vae/diffusion_pytorch_model.bin','safety_checker/config.json','safety_checker/pytorch_model.bin','feature_extractor/*'],max_workers=2)
(BASE/'records/gaussianeditor_sd15.json').write_text(json.dumps({'repo':repo,'revision':revision,'path':path},indent=2));print('SD15 ready',flush=True)
