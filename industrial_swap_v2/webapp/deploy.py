"""Copy application source only; never upload data, environments, or secrets."""
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parents[2]
base=root/'industrial_swap_v2'; archive=base/'tmp/webapp_code.tar'
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive,'w') as tar:
    for path in (base/'webapp').rglob('*'):
        relative=path.relative_to(base/'webapp')
        if any(part in ('data','__pycache__') for part in relative.parts) or not path.is_file(): continue
        tar.add(path,arcname=path.relative_to(root).as_posix())
    for name in ('core/context.py','core/qwen_client.py','modules/m2_image_generation.py','tests/test_qwen_once.py'):
        path=base/name; tar.add(path,arcname=path.relative_to(root).as_posix())
options=['-o','BatchMode=yes','-o','ClearAllForwardings=yes']
remote='/mnt/sda/ubuntu/lkx/papers/paper_eli'
subprocess.run(['scp',*options,str(archive),f'ubuntuA6000:{remote}/industrial_swap_v2/tmp/'],check=True)
subprocess.run(['ssh',*options,'ubuntuA6000',f'cd {remote} && tar -xf industrial_swap_v2/tmp/webapp_code.tar'],check=True)
