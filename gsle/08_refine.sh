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
test -f "${GSLE_OUTPUT:-$ROOT/outputs/gsle}/composition/scene.ply"
envs/gsle/bin/python -u gsle/train_object.py refine "$@"
envs/gsle/bin/python -u gsle/compose.py refine
envs/gsle/bin/python -u gsle/render_results.py refine
envs/gsle/bin/python -u gsle/review_results.py
envs/gsle/bin/python -u gsle/verify_artifacts.py
