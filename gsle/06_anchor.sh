set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
envs/gsle/bin/python -u gsle/prepare_edit_region.py
envs/gsle/bin/python -u gsle/avp.py
bash gsle/06_generate.sh
