set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
CONDA=/home/ubuntu/anaconda3/bin/conda
if ! test -x "$ROOT/envs/gaussianeditor/bin/python"; then
  "$CONDA" create -y -p "$ROOT/envs/gaussianeditor" --override-channels -c conda-forge python=3.8 pip
fi
export PATH="$ROOT/envs/gaussianeditor/bin:$PATH"
python -m pip install 'pip<25' 'setuptools<70' wheel ninja poetry-core==1.9.1
python "$ROOT/scripts/download_torch_wheel.py"
python -m pip install "$ROOT/runtime/wheels/torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl" torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
cd "$ROOT/repos/GaussianEditor"
"$ROOT/envs/download/bin/python" "$ROOT/scripts/prepare_ge_dependencies.py"
"$ROOT/envs/download/bin/python" "$ROOT/scripts/02a_pin_transitive.py"
python -m pip install --no-build-isolation --no-clean -c "$ROOT/scripts/ge_constraints.txt" -r "$ROOT/records/ge_requirements_local.txt"
python -m pip install pytorch-lightning==2.0.9 'numpy<1.25' lpips
python -m pip freeze > "$ROOT/records/gaussianeditor.pip.freeze.txt"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); import threestudio; print("GaussianEditor import OK")'
date -Iseconds > "$ROOT/records/gaussianeditor.environment.completed"
