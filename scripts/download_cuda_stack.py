"""Download torch's exact declared Linux CUDA dependencies with verified hashes."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile
from email.parser import BytesParser
from packaging.requirements import Requirement

root=Path(os.environ['ROOT'])
directory=root/'runtime/wheels'
wheel=directory/'torch-2.4.1+cu118-cp39-cp39-linux_x86_64.whl'
subprocess.run([str(root/'envs/download/bin/python'),str(root/'scripts/download_torch_wheel.py'),'2.4.1','cp39'],check=True)
with zipfile.ZipFile(wheel) as archive:
    metadata=BytesParser().parsebytes(archive.read(next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))))
files=[str(wheel)]
records=[]
for line in metadata.get_all('Requires-Dist',[]):
    requirement=Requirement(line)
    if not (requirement.name.startswith('nvidia-') or requirement.name=='triton'):
        continue
    version=next(spec.version for spec in requirement.specifier if spec.operator=='==')
    with urllib.request.urlopen(f'https://pypi.org/pypi/{requirement.name}/{version}/json',timeout=60) as response:
        info=json.load(response)
    candidates=[item for item in info['urls'] if item['filename'].endswith('.whl') and 'x86_64' in item['filename'] and 'linux' in item['filename'] and ('-py3-' in item['filename'] or '-cp39-cp39-' in item['filename'])]
    assert candidates, requirement.name
    item=candidates[0]
    filename=item['filename']
    print(f'Downloading {filename} ({item["size"]} bytes)',flush=True)
    subprocess.run(['aria2c','--continue=true','--auto-file-renaming=false','--max-connection-per-server=8','--split=8','--min-split-size=32M','--file-allocation=none','--summary-interval=30','--console-log-level=warn',f'--all-proxy={os.environ["https_proxy"]}',f'--checksum=sha-256={item["digests"]["sha256"]}',f'--dir={directory}',f'--out={filename}',item['url']],check=True)
    files.append(str(directory/filename))
    records.append({'package':requirement.name,'version':version,'filename':filename,'sha256':item['digests']['sha256'],'url':item['url']})
(root/'records/editsplat_cuda_wheels.txt').write_text('\n'.join(files)+'\n')
(root/'records/editsplat_cuda_wheels.json').write_text(json.dumps(records,indent=2))
