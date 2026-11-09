set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
test -f records/gsle/language.completed
envs/gsle_langsplat/bin/python -u gsle/reference_proposals.py
envs/gsle_langsplat/bin/python -u gsle/query_language.py
envs/gsle/bin/python -u gsle/localize.py
envs/gsle/bin/python -u gsle/preview_localization.py
