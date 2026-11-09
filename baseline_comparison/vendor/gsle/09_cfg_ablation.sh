set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
envs/download/bin/python gsle/prepare_cfg_ablation.py
export GSLE_OUTPUT="$ROOT/outputs/gsle/ablations/text_only_cfg"
export GSLE_RECORDS="$ROOT/records/gsle/ablations/text_only_cfg"
export GSLE_CONFIG="$GSLE_OUTPUT/config.json"
bash gsle/08_refine.sh "$@"
