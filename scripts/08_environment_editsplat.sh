set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
CONDA=/home/ubuntu/anaconda3/bin/conda
if ! test -x "$ROOT/envs/editsplat/bin/python"; then
  "$CONDA" create -y -p "$ROOT/envs/editsplat" --override-channels -c conda-forge python=3.9 pip
fi
export PATH="$ROOT/envs/editsplat/bin:$PATH"
python -m pip install 'pip<25' 'setuptools<70' wheel ninja poetry-core==1.9.1
"$ROOT/envs/download/bin/python" "$ROOT/scripts/download_cuda_stack.py"
python -m pip install -r "$ROOT/records/editsplat_cuda_wheels.txt" torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu118
cd "$ROOT/repos/cvpr25_EditSplat"
export PIP_CONSTRAINT="$ROOT/scripts/editsplat_constraints.txt"
python -m pip install diffusers==0.30.3 image-reward==1.5 plyfile scikit-learn tqdm 'opencv-python<4.12' einops scikit-image matplotlib lpips numpy==1.26.1 jaxtyping 'huggingface-hub<1' 'transformers<4.46' 'accelerate<1'
"$ROOT/envs/download/bin/python" "$ROOT/scripts/prepare_editsplat_dependencies.py"
python -m pip install --no-build-isolation "$ROOT/repos/dependencies/lang-sam-editsplat" git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
python -c 'import torch, diffusers, lang_sam, ImageReward, diff_gaussian_rasterization; print(torch.__version__, diffusers.__version__); assert torch.__version__.split("+")[0] == "2.4.1"'
python -m pip freeze > "$ROOT/records/editsplat.pip.freeze.txt"
date -Iseconds > "$ROOT/records/editsplat.environment.completed"
