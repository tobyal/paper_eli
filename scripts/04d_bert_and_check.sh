set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
envs/download/bin/python scripts/download_hf_parallel.py bert_safe
envs/editsplat/bin/python scripts/08c_check_editsplat_models.py
