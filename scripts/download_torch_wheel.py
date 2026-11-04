"""Download the upstream wheel with parallel ranges and its published SHA256."""
import html
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

root = Path(os.environ['ROOT'])
version = sys.argv[1] if len(sys.argv)>1 else '2.1.0'
python_tag = sys.argv[2] if len(sys.argv)>2 else 'cp38'
directory = root/'runtime/wheels'
directory.mkdir(parents=True, exist_ok=True)
index = 'https://download.pytorch.org/whl/cu118/torch/'
document = urllib.request.urlopen(index, timeout=60).read().decode()
links = [html.unescape(x) for x in re.findall(r'href="([^"]+)"', document)]
wheel = f'torch-{version}+cu118-{python_tag}-{python_tag}-linux_x86_64.whl'
link = next(x for x in links if wheel in urllib.parse.unquote(x))
url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(index, link))
digest = urllib.parse.parse_qs(fragment)['sha256'][0]
filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.split('/')[-1])
(root/f'records/torch_{version}_{python_tag}.sha256').write_text(f'{digest}  {filename}\n')
target = directory/filename
if target.exists():
    checksum = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(block)
    if checksum.hexdigest() == digest:
        print(f'Existing wheel verified: {filename}', flush=True)
        sys.exit(0)
subprocess.run(['aria2c', '--allow-overwrite=true', '--auto-file-renaming=false', '--continue=true', '--max-connection-per-server=8', '--split=8', '--min-split-size=32M', '--file-allocation=none', '--summary-interval=30', '--console-log-level=warn', '--max-tries=10', '--retry-wait=3', f'--all-proxy={os.environ["https_proxy"]}', f'--checksum=sha-256={digest}', f'--dir={directory}', f'--out={filename}', url], check=True)
