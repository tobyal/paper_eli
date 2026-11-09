"""Resolve moved upstream model namespace and official Omnidata archive."""
import os,json,hashlib,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi,snapshot_download
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
def variations():
 repo='lambda/sd-image-variations-diffusers';info=HfApi().model_info(repo)
 patterns=['image_encoder/*','feature_extractor/*','vae/*','scheduler/*','model_index.json']
 p=Path(snapshot_download(repo,revision=info.sha,allow_patterns=patterns,ignore_patterns=['*.safetensors','*.onnx','*.msgpack'],max_workers=2))
 assert (p/'image_encoder/pytorch_model.bin').is_file()
 assert (p/'vae/diffusion_pytorch_model.bin').is_file()
 return repo,{'path':str(p),'revision':info.sha}
def depth():
 url='https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt?download=1'
 p=ROOT/'models/omnidata/omnidata_dpt_depth_v2.ckpt';p.parent.mkdir(exist_ok=True)
 if not p.exists():
  part=p.with_suffix('.part')
  with requests.get(url,stream=True,timeout=(30,300)) as resp:
   resp.raise_for_status()
   with part.open('wb') as f:
    for chunk in resp.iter_content(8*1024*1024):f.write(chunk)
  part.rename(p)
 md5=hashlib.md5();sha=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):md5.update(chunk);sha.update(chunk)
 assert md5.hexdigest()=='1071266b8bfdb58cdcb490d6fcddedb0'
 return 'omnidata',{'path':str(p),'url':url,'md5':md5.hexdigest(),'sha256':sha.hexdigest()}
record={}
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(f) for f in [variations,depth]]
 for fut in futures:
  key,value=fut.result();record[key]=value;print('READY',key,flush=True)
  (BASE/'records/gaussianeditor_recovered_models.json').write_text(json.dumps(record,indent=2))
