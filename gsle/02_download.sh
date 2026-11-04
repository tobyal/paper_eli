set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
envs/download/bin/python gsle/resolve_models.py
envs/download/bin/python gsle/download_models.py
