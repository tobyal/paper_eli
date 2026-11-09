"""Fetch only assets required by the upstream GaussianEditor Delete/Add path."""
import os,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison'
api=HfApi();dest=BASE/'records';dest.mkdir(exist_ok=True)
requests={
 'camenduru/Wonder3D':['unet/*'],
 'lambdalabs/sd-image-variations-diffusers':['image_encoder/*','feature_extractor/*','vae/*','scheduler/*','model_index.json'],
 'lllyasviel/control_v11p_sd15_inpaint':['config.json','diffusion_pytorch_model.bin'],
 'lllyasviel/Annotators':['omnidata_dpt_depth_v2.ckpt'],
}
def one(repo,patterns):
 info=api.model_info(repo);files=[s.rfilename for s in info.siblings]
 if repo.endswith('/Annotators') and patterns[0] not in files:raise ValueError('DPT checkpoint absent in requested model repository')
 path=snapshot_download(repo,revision=info.sha,allow_patterns=patterns,ignore_patterns=['*.safetensors','*.onnx','*.msgpack'],max_workers=2)
 print('READY',repo,path,flush=True)
 return repo,{'revision':info.sha,'path':path,'allow_patterns':patterns}
record={}
with ThreadPoolExecutor(max_workers=3) as pool:
 jobs={pool.submit(one,k,v):k for k,v in requests.items()}
 for future in as_completed(jobs):
  key=jobs[future]
  try:k,v=future.result();record[k]=v
  except Exception as exc:record[key]={'error':str(exc)};print('FAILED',key,str(exc),flush=True)
  (dest/'gaussianeditor_add_models.json').write_text(json.dumps(record,indent=2))
if any('error' in v for v in record.values()):raise SystemExit(1)
