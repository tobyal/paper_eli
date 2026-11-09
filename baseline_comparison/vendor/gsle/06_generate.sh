set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
test -f outputs/gsle/avp/anchor.json
envs/gsle/bin/python -u gsle/inpaint.py
envs/gsle_langsplat/bin/python -u gsle/rank_inpainting.py
envs/gsle_langsplat/bin/python -u gsle/sam_foreground.py
