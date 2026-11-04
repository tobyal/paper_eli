set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
for arg in "$@"; do
  if [[ "$arg" == --steps* ]]; then
    echo 'Use train_object.py directly for diagnostic steps; this wrapper composes final results.' >&2
    exit 2
  fi
done
envs/gsle/bin/python -u gsle/train_object.py coarse "$@"
envs/gsle/bin/python -u gsle/compose.py coarse
envs/gsle/bin/python -u gsle/render_results.py coarse --preview
