#!/usr/bin/env bash
set -euo pipefail
V2=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=$(dirname "$V2")
export http_proxy=http://127.0.0.1:17891 https_proxy=http://127.0.0.1:17891
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" NO_PROXY=localhost,127.0.0.1
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export TMPDIR="$V2/tmp" PIP_CACHE_DIR="$V2/models/pip" TORCH_EXTENSIONS_DIR="$V2/tmp/torch_extensions"
export HF_HOME="$V2/models/huggingface" HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$V2/records" "$V2/logs" "$V2/third_party"
if ! test -x "$V2/envs/trellis2/bin/python"; then
  "$PROJECT/envs/download/bin/python" -m venv "$V2/envs/trellis2"
fi
export PATH="$V2/envs/trellis2/bin:$CUDA_HOME/bin:$PATH"
python -m pip install --upgrade 'pip<26' setuptools wheel packaging ninja
python -m pip install triton==3.2.0
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless 'trimesh==4.11.3' 'transformers==4.57.6' 'gradio==6.0.1' tensorboard pandas lpips zstandard pillow kornia timm scipy plyfile pyyaml requests
python -m pip install 'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'
python -m pip install flash-attn==2.7.3 --no-build-isolation
clone() {
  local name="$1" url="$2" branch="$3"
  if ! test -d "$V2/third_party/$name/.git"; then
    git clone --recursive --branch "$branch" "$url" "$V2/third_party/$name"
  fi
}
clone TRELLIS.2 https://github.com/microsoft/TRELLIS.2.git main
git -C "$V2/third_party/TRELLIS.2" checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$V2/third_party/TRELLIS.2" submodule update --init --recursive
clone nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0
clone nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git renderutils
clone CuMesh https://github.com/JeffreyXiang/CuMesh.git main
clone FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git main
for package in nvdiffrast nvdiffrec CuMesh FlexGEMM; do
  python -m pip install "$V2/third_party/$package" --no-build-isolation
done
python -m pip install "$V2/third_party/TRELLIS.2/o-voxel" --no-build-isolation
python -m pip freeze > "$V2/records/trellis2.pip.freeze.txt"
cd "$PROJECT"
python -m industrial_swap_v2.scripts.record_environment
