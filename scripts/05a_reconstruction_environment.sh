set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
if ! test -x envs/reconstruction/bin/python; then
  envs/gaussianeditor/bin/python -m venv --system-site-packages envs/reconstruction
fi
export PATH="$ROOT/envs/reconstruction/bin:$PATH"
python -m pip install 'numpy==1.24.4' 'Pillow==9.5.0' 'plyfile==1.0.2' tqdm ninja 'setuptools<70' wheel
python -m pip install --no-build-isolation "$ROOT/repos/gaussian-splatting/submodules/diff-gaussian-rasterization" "$ROOT/repos/gaussian-splatting/submodules/simple-knn"
python -m pip freeze > records/reconstruction.pip.freeze.txt
python -c 'import torch, diff_gaussian_rasterization, simple_knn; print(torch.__version__); assert torch.cuda.is_available()'
date -Iseconds > records/reconstruction.environment.completed
