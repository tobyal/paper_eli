"""Windows-to-SSH code sync; excludes results, credentials and active shell launchers."""
from pathlib import Path
import subprocess
import tarfile
root=Path(__file__).resolve().parents[2]
base=root/'industrial_swap_v2'
archive=base/'tmp/code_update.tar'
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive,'w') as tar:
    for folder in ('core','modules','scripts','tests'):
        for path in (base/folder).glob('*.py'):
            tar.add(path,arcname=path.relative_to(root).as_posix())
    # Shell launchers are deliberately excluded: updating a running Bash script
    # can make its next command resume at a changed byte offset. Deploy those
    # files through Git or an explicit copy after verifying they are inactive.
    for name in ('run.py','configs/default.yaml','README.md','.gitignore'):
        path=base/name
        if path.is_file(): tar.add(path,arcname=path.relative_to(root).as_posix())
remote='/mnt/sda/ubuntu/lkx/papers/paper_eli'
options=['-o','BatchMode=yes','-o','ClearAllForwardings=yes']
subprocess.run(['scp',*options,str(archive),f'ubuntuA6000:{remote}/industrial_swap_v2/tmp/'],check=True)
subprocess.run(['ssh',*options,'ubuntuA6000',f'cd {remote} && tar -xf industrial_swap_v2/tmp/code_update.tar'],check=True)
