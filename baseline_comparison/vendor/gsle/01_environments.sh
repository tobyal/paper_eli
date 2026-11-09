set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
test -f records/gsle/upstream.json
if ! test -x envs/gsle_langsplat/bin/python; then
  envs/gaussianeditor/bin/python -m venv --system-site-packages envs/gsle_langsplat
fi
export PATH="$ROOT/envs/gsle_langsplat/bin:$CUDA_HOME/bin:$PATH"
python -m pip install --no-deps open-clip-torch==2.24.0
python -m pip install --no-build-isolation --no-deps repos/segment-anything-langsplat repos/LangSplat/submodules/langsplat-rasterization
python -c 'import torch,open_clip,segment_anything; from diff_gaussian_rasterization import GaussianRasterizer; print(torch.__version__,segment_anything.__file__)'
python -m pip freeze > records/gsle/langsplat.pip.freeze.txt
date -Is > records/gsle/langsplat.environment.completed
if ! test -x envs/gsle/bin/python; then
  envs/editsplat/bin/python -m venv --system-site-packages envs/gsle
fi
export PATH="$ROOT/envs/gsle/bin:$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation --no-deps repos/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install matplotlib==3.9.4
python -c 'import torch,diffusers; from diff_gaussian_rasterization import GaussianRasterizer; print(torch.__version__,diffusers.__version__)'
python -m pip freeze > records/gsle/gsle.pip.freeze.txt
date -Is > records/gsle/gsle.environment.completed
