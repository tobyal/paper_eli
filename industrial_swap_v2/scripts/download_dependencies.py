"""Download pinned official TRELLIS dependency weights, no generation calls."""
import json
import os
import argparse
from pathlib import Path
base=Path(__file__).resolve().parents[1]
# Capture existing authorized HF credential before selecting the isolated cache.
from huggingface_hub import get_token, HfApi, hf_hub_download, snapshot_download
token=get_token()
p=argparse.ArgumentParser()
p.add_argument('--token-file',type=Path)
args=p.parse_args()
if args.token_file:
    token=args.token_file.read_text(encoding='utf-8-sig').strip()
    if not token.startswith('hf_') or any(c.isspace() for c in token):
        raise ValueError('Expected one complete Hugging Face token in the protected file')
cache=base/'models/huggingface/hub'
os.environ['HF_HUB_DISABLE_XET']='1'
records={}
revisions={
    'microsoft/TRELLIS-image-large':'25e0d31ffbebe4b5a97464dd851910efc3002d96',
    'facebook/dinov3-vitl16-pretrain-lvd1689m':'ea8dc2863c51be0a264bab82070e3e8836b02d51',
    'briaai/RMBG-2.0':'5df4c9c76d8170882c34f6986e848ee07fd0ba43'}
for repo, patterns in [
    ('microsoft/TRELLIS-image-large',['ckpts/ss_dec_conv3d_16l8_fp16.*']),
    ('facebook/dinov3-vitl16-pretrain-lvd1689m',['config.json','*.safetensors','*.json']),
    ('briaai/RMBG-2.0',['*.json','*.py','*.safetensors'])]:
    try:
        info=HfApi(token=token).model_info(repo,revision=revisions[repo])
        records[repo]={'revision':info.sha,'gated':info.gated,'status':'downloading'}
        (base/'records/trellis_dependencies.json').write_text(json.dumps(records,indent=2))
        path=snapshot_download(repo,revision=info.sha,allow_patterns=patterns,cache_dir=cache,token=token,max_workers=2)
        records[repo].update(status='complete',snapshot_path=path)
    except Exception as error:
        records.setdefault(repo,{}).update(status='failed',error_type=type(error).__name__)
        print(repo,type(error).__name__,flush=True)
    (base/'records/trellis_dependencies.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records),flush=True)
