set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
export PATH="$ROOT/envs/dreamcatalyst/bin:$PATH"
cd "$ROOT/repos/DreamCatalyst/threestudio"
python - <<'PY'
import os
from pathlib import Path
import subprocess
root=Path(os.environ['ROOT'])
path=Path('gaussiansplatting/submodules/diff-gaussian-rasterization/setup.py')
text=path.read_text()
if 'extracompileargs=' in text:
    assert text.count('extracompileargs=')==1
    path.write_text(text.replace('extracompileargs=','extra_compile_args='))
(root/'records/DreamCatalyst_build_keyword.patch').write_text(subprocess.check_output(['git','diff','--',str(path)],text=True))
PY
python -m pip install --no-build-isolation --force-reinstall --no-deps ./gaussiansplatting/submodules/diff-gaussian-rasterization ./gaussiansplatting/submodules/simple-knn "$ROOT/repos/dependencies/GroundingDINO"
python -c 'import torch, threestudio, groundingdino._C; print(torch.__version__); assert torch.__version__.split("+")[0]=="2.0.1"; print("DreamCatalyst imports and matching CUDA kernels OK")'
python -m pip freeze > "$ROOT/records/dreamcatalyst.pip.freeze.txt"
date -Iseconds > "$ROOT/records/dreamcatalyst.environment.completed"
