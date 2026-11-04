"""Fetch exact upstream files into the standard HF cache with verified ranges."""
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata

root=Path(os.environ['ROOT'])
group=sys.argv[1] if len(sys.argv)>1 else 'core'
groups={
 'core':{
  'timbrooks/instruct-pix2pix':['model_index.json','scheduler/*','tokenizer/*','text_encoder/config.json','text_encoder/pytorch_model.bin','unet/config.json','unet/diffusion_pytorch_model.bin','vae/config.json','vae/diffusion_pytorch_model.bin'],
  'stable-diffusion-v1-5/stable-diffusion-v1-5':['tokenizer/*','text_encoder/config.json','text_encoder/pytorch_model.bin'],
  'CompVis/stable-diffusion-v1-4':['scheduler/*'],
 },
 'aux':{
  'ShilongLiu/GroundingDINO':['GroundingDINO_SwinB.cfg.py','groundingdino_swinb_cogcoor.pth','GroundingDINO_SwinT_OGC.cfg.py','groundingdino_swint_ogc.pth'],
  'google-bert/bert-base-uncased':['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','pytorch_model.bin'],
 },
 'editsplat_safety':{
  'timbrooks/instruct-pix2pix':['feature_extractor/*','safety_checker/config.json','safety_checker/pytorch_model.bin'],
 },
 'imagereward':{
  'THUDM/ImageReward':['ImageReward.pt','med_config.json'],
 },
 'bert_safe':{
  'google-bert/bert-base-uncased':['model.safetensors'],
 },
}
revisions={
 'timbrooks/instruct-pix2pix':'31519b5cb02a7fd89b906d88731cd4d6a7bbf88d',
 'stable-diffusion-v1-5/stable-diffusion-v1-5':'451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
 'CompVis/stable-diffusion-v1-4':'133a221b8aa7292a167afc5127cb63fb5005638b',
 'ShilongLiu/GroundingDINO':'a94c9b567a2a374598f05c584e96798a170c56fb',
 'google-bert/bert-base-uncased':'86b5e0934494bd15c9632b12f734a8a67f723594',
 'THUDM/ImageReward':'5736be03b2652728fb87788c9797b0570450ab72',
}
record={}
for repo,patterns in groups[group].items():
    info=HfApi().model_info(repo,revision=revisions[repo],files_metadata=True)
    cache=Path(os.environ['HF_HOME'])/'hub'/('models--'+repo.replace('/','--'))
    files=[]
    for entry in info.siblings:
        filename=entry.rfilename
        if not any(fnmatch.fnmatch(filename,p) for p in patterns):
            continue
        if entry.size and entry.size>50*1024*1024:
            metadata=get_hf_file_metadata(hf_hub_url(repo,filename,revision=info.sha))
            assert len(metadata.etag)==64, 'Large LFS asset must have SHA256 etag'
            blob=cache/'blobs'/metadata.etag
            blob.parent.mkdir(parents=True,exist_ok=True)
            if not blob.exists():
                temporary=blob.with_name(blob.name+'.incomplete')
                print(f'{repo}/{filename}: {entry.size} bytes, SHA256 {metadata.etag}',flush=True)
                command=['aria2c','--continue=true','--auto-file-renaming=false','--max-connection-per-server=8','--split=8','--min-split-size=32M','--file-allocation=none','--summary-interval=30','--console-log-level=warn','--max-tries=10','--retry-wait=3',f'--all-proxy={os.environ["https_proxy"]}',f'--checksum=sha-256={metadata.etag}',f'--dir={blob.parent}',f'--out={temporary.name}',hf_hub_url(repo,filename,revision=info.sha)]
                subprocess.run(command,check=True)
                temporary.replace(blob)
            link=cache/'snapshots'/info.sha/filename
            link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():
                link.symlink_to(os.path.relpath(blob,link.parent))
            files.append({'name':filename,'size':entry.size,'sha256':metadata.etag})
        else:
            hf_hub_download(repo,filename,revision=info.sha)
            files.append({'name':filename,'size':entry.size})
    ref=cache/'refs/main'
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(info.sha)
    record[repo]={'revision':info.sha,'path':str(cache/'snapshots'/info.sha),'files':files}
    (root/f'records/{group}_model_downloads.json').write_text(json.dumps(record,indent=2))
print(f'{group} model files verified',flush=True)
