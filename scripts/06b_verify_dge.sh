set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT/repos/DGE"
"$ROOT/envs/download/bin/python" - <<'PY'
import os
from pathlib import Path
import subprocess
root=Path(os.environ['ROOT'])
path=Path('threestudio/systems/DGE.py')
text=path.read_text()
before='field(default_factory=[999, 200, 200, 21])'
after='field(default_factory=lambda: [999, 200, 200, 21])'
if after not in text:
    assert text.count(before)==1
    path.write_text(text.replace(before,after))
(root/'records/DGE_dataclass_compatibility.patch').write_text(subprocess.check_output(['git','diff','--',str(path)],text=True))
PY
"$ROOT/envs/dge/bin/python" -c 'import torch, threestudio; from threestudio.systems.DGE import DGE; config=DGE.Config(); print(torch.__version__, config.added_noise_schedule); print("DGE imports and dataclass defaults OK")'
"$ROOT/envs/dge/bin/python" -m pip freeze > "$ROOT/records/dge.pip.freeze.txt"
date -Iseconds > "$ROOT/records/dge.environment.completed"
