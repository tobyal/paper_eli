#!/usr/bin/env bash
# Reuse verified data and environments; create independent outputs and caches.
set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
tag="${1:?Usage: bash scripts/replay_baseline.sh NEW_OUTPUT_TAG}"
for method in gaussianeditor dge dreamcatalyst editsplat; do
  test -f "records/$method.environment.completed"
  test ! -e "outputs/${method}_${tag}"
done
for method in gaussianeditor dge dreamcatalyst editsplat; do
  envs/download/bin/python scripts/run_method.py "$method" --output-tag "$tag"
  envs/download/bin/python scripts/finish_method.py "$method"
done
envs/editsplat/bin/python scripts/evaluate_common.py --methods "gaussianeditor_$tag" "dge_$tag" "dreamcatalyst_$tag" "editsplat_$tag"
cp records/common_metrics.json "records/replay_${tag}_metrics.json"
printf 'Replay complete. Inspect actual RGB outputs and metrics for semantic success.\n'
