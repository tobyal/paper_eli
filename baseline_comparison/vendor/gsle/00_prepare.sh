set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
mkdir -p gsle records/gsle logs/gsle outputs/gsle references/gsle
envs/download/bin/python gsle/bootstrap_sources.py
