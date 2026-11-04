set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
test -f "$ROOT/records/gaussianeditor.environment.completed"
if ! test -x "$ROOT/envs/dreamcatalyst/bin/python"; then
  /home/ubuntu/anaconda3/bin/conda create -y -p "$ROOT/envs/dreamcatalyst" --clone "$ROOT/envs/gaussianeditor"
fi
export PATH="$ROOT/envs/dreamcatalyst/bin:$PATH"
cd "$ROOT/repos/DreamCatalyst/threestudio"
python "$ROOT/scripts/download_torch_wheel.py" 2.0.1 cp38
python -m pip install "$ROOT/runtime/wheels/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl" torchvision==0.15.2 torchaudio==2.0.2 xformers==0.0.22 --extra-index-url https://download.pytorch.org/whl/cu118
# README explicitly requests diffusers 0.27.2; requirements_all lists 0.19.3.
# Honor the README and retain a hub version with cached_download for compatibility.
python -m pip install diffusers==0.27.2 transformers==4.42.4 huggingface-hub==0.25.2 accelerate==0.30.1
python -m pip install --no-build-isolation --force-reinstall --no-deps "$ROOT/repos/dependencies/tiny-cuda-nn/bindings/torch" "$ROOT/repos/dependencies/nvdiffrast" "$ROOT/repos/dependencies/nerfacc"
bash "$ROOT/scripts/07b_finish_dreamcatalyst.sh"
