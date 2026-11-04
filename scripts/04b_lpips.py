"""Download the exact LPIPS checkpoint required by the official implementation."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

root=Path(os.environ['ROOT'])
target=root/'models/lpips/vgg.pth'
target.parent.mkdir(parents=True,exist_ok=True)
urls=['https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1',
      'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth']
expected='d507d7349b931f0638a25a48a722f98a'
for url in urls:
    try:
        payload=urllib.request.urlopen(url,timeout=30).read()
        if hashlib.md5(payload).hexdigest()!=expected:
            raise ValueError('Checkpoint differs from author-published MD5')
        target.write_bytes(payload)
        (root/'records/lpips_checkpoint.json').write_text(json.dumps({'url':url,'md5':expected,'sha256':hashlib.sha256(payload).hexdigest()},indent=2))
        break
    except Exception as error:
        print(f'{url}: {error}',flush=True)
else:
    raise RuntimeError('Could not obtain exact official LPIPS checkpoint')
for repo in ['GaussianEditor','DGE','DreamCatalyst/threestudio']:
    link=root/'repos'/repo/'threestudio/utils/lpips/vgg.pth'
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists(): link.symlink_to(target)
print('LPIPS checkpoint verified and linked',flush=True)
