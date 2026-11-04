set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
export PATH="$ROOT/envs/editsplat/bin:$PATH"
export PIP_CONSTRAINT="$ROOT/scripts/editsplat_constraints.txt"
cd "$ROOT/repos/cvpr25_EditSplat"
python -m pip install poetry-core==1.9.1
"$ROOT/envs/download/bin/python" "$ROOT/scripts/prepare_editsplat_dependencies.py"
python -m pip install --no-build-isolation "$ROOT/repos/dependencies/lang-sam-editsplat" git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
python -c 'import torch, diffusers, lang_sam, ImageReward, diff_gaussian_rasterization; print(torch.__version__, diffusers.__version__); assert torch.__version__.split("+")[0] == "2.4.1"'
python -m pip freeze > "$ROOT/records/editsplat.pip.freeze.txt"
git -C "$ROOT/repos/dependencies/lang-sam-editsplat" rev-parse HEAD > "$ROOT/records/editsplat_langsam_commit.txt"
date -Iseconds > "$ROOT/records/editsplat.environment.completed"
