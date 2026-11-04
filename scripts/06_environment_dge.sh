set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
test -f "$ROOT/records/gaussianeditor.environment.completed"
if ! test -x "$ROOT/envs/dge/bin/python"; then
  /home/ubuntu/anaconda3/bin/conda create -y -p "$ROOT/envs/dge" --clone "$ROOT/envs/gaussianeditor"
fi
export PATH="$ROOT/envs/dge/bin:$PATH"
cd "$ROOT/repos/DGE"
# DGE accepts torch>=2; use the identical pinned GE stack, then its own kernels.
python -m pip install --no-build-isolation --force-reinstall --no-deps ./gaussiansplatting/submodules/diff-gaussian-rasterization ./gaussiansplatting/submodules/simple-knn
python -c 'import torch, threestudio; print(torch.__version__); print("DGE import OK")'
python -m pip freeze > "$ROOT/records/dge.pip.freeze.txt"
date -Iseconds > "$ROOT/records/dge.environment.completed"
