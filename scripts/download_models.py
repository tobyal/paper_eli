import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ['ROOT'])
models = {
    'timbrooks/instruct-pix2pix': ['model_index.json', 'scheduler/*', 'tokenizer/*', 'text_encoder/config.json', 'text_encoder/pytorch_model.bin', 'unet/config.json', 'unet/diffusion_pytorch_model.bin', 'vae/config.json', 'vae/diffusion_pytorch_model.bin'],
    'stable-diffusion-v1-5/stable-diffusion-v1-5': ['tokenizer/*', 'text_encoder/config.json', 'text_encoder/pytorch_model.bin'],
    'CompVis/stable-diffusion-v1-4': ['scheduler/*'],
    'ShilongLiu/GroundingDINO': ['GroundingDINO_SwinT_OGC.cfg.py', 'groundingdino_swint_ogc.pth'],
}
record = {}
api = HfApi()
for repo_id, patterns in models.items():
    info = api.model_info(repo_id)
    print(f'Downloading {repo_id} @ {info.sha}', flush=True)
    path = snapshot_download(repo_id, revision=info.sha, allow_patterns=patterns, max_workers=3)
    # Pin main to this exact snapshot for older diffusers/hub clients.
    ref = Path(path).parent.parent / 'refs' / 'main'
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(info.sha)
    record[repo_id] = {'revision': info.sha, 'path': path, 'patterns': patterns}
    (root/'records/model_downloads.json').write_text(json.dumps(record, indent=2))
print('Model download stage complete', flush=True)
