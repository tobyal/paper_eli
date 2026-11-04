import hashlib
import json
import os
from pathlib import Path
import subprocess
from huggingface_hub import HfApi, snapshot_download

root=Path(os.environ['ROOT'])
record={}
revisions={'ShilongLiu/GroundingDINO':'a94c9b567a2a374598f05c584e96798a170c56fb','google-bert/bert-base-uncased':'86b5e0934494bd15c9632b12f734a8a67f723594'}
for repo,patterns in {
    'ShilongLiu/GroundingDINO':['GroundingDINO_SwinB.cfg.py','groundingdino_swinb_cogcoor.pth'],
    'google-bert/bert-base-uncased':['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','pytorch_model.bin'],
}.items():
    sha=revisions[repo]
    path=snapshot_download(repo,revision=sha,allow_patterns=patterns,max_workers=2)
    ref=Path(path).parent.parent/'refs/main'
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(sha)
    record[repo]={'revision':sha,'path':path}
    (root/'records/aux_model_downloads.json').write_text(json.dumps(record,indent=2))

target=root/'models/torch/hub/checkpoints'
target.mkdir(parents=True,exist_ok=True)
checksums={'sam_vit_h_4b8939.pth':'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e','vgg16-397923af.pth':'397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}
for name,url in {
    'sam_vit_h_4b8939.pth':'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vgg16-397923af.pth':'https://download.pytorch.org/models/vgg16-397923af.pth',
}.items():
    subprocess.run(['aria2c','--continue=true','--auto-file-renaming=false','--max-connection-per-server=8','--split=8','--min-split-size=32M','--file-allocation=none','--summary-interval=30','--console-log-level=warn',f'--all-proxy={os.environ["https_proxy"]}',f'--dir={target}',f'--out={name}',url],check=True)
    digest=hashlib.sha256()
    with (target/name).open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):
            digest.update(chunk)
    assert digest.hexdigest()==checksums[name], name+' checksum mismatch'
    record[name]={'url':url,'sha256':digest.hexdigest(),'bytes':(target/name).stat().st_size}
    (root/'records/aux_model_downloads.json').write_text(json.dumps(record,indent=2))

alias=Path(os.environ['HF_HOME'])/'hub/models--bert-base-uncased'
if not alias.exists(): alias.symlink_to('models--google-bert--bert-base-uncased',target_is_directory=True)
