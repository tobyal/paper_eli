"""Verify name-based legacy loaders resolve the same pinned cached models offline."""
import json,os
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(os.environ['ROOT']);checked={}
for repo,revision,files in [
 ('ShilongLiu/GroundingDINO','a94c9b567a2a374598f05c584e96798a170c56fb',['GroundingDINO_SwinB.cfg.py','groundingdino_swinb_cogcoor.pth','GroundingDINO_SwinT_OGC.cfg.py','groundingdino_swint_ogc.pth']),
 ('bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594',['config.json','pytorch_model.bin','model.safetensors','tokenizer_config.json','vocab.txt'])]:
 for name in files:
  path=hf_hub_download(repo,name,local_files_only=True)
  assert revision in path,path
  checked[repo+'/'+name]=path
for name in ['sam_vit_h_4b8939.pth','vgg16-397923af.pth']:
 path=root/'models/torch/hub/checkpoints'/name
 assert path.is_file() and path.stat().st_size>100000000
 checked[name]=str(path)
(root/'records/offline_cache_verified.json').write_text(json.dumps(checked,indent=2))
print(f'Validated {len(checked)} legacy model assets offline; direct local diffusion paths are recorded separately')
