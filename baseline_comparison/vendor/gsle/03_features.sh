set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
if [[ "${GSLE_DATASET_PREPARED:-0}" != 1 ]]; then
  envs/gsle_langsplat/bin/python gsle/prepare_dataset.py
fi
envs/gsle_langsplat/bin/python -u gsle/lang_features.py "$@"
