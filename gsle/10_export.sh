set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
stage="${1:-refine}"
case "$stage" in
  coarse|refine) ;;
  *) echo 'Expected coarse or refine' >&2; exit 2 ;;
esac
test -f "${GSLE_OUTPUT:-$ROOT/outputs/gsle}/$stage/final_summary.json"
envs/gsle/bin/python -u gsle/compose.py "$stage"
if [[ "$stage" == coarse ]]; then
  envs/gsle/bin/python -u gsle/render_results.py coarse --preview
else
  envs/gsle/bin/python -u gsle/render_results.py refine
  envs/gsle/bin/python -u gsle/review_results.py
  envs/gsle/bin/python -u gsle/verify_artifacts.py
fi
