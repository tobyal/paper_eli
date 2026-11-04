set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
deadline=$((SECONDS + 3600))
until test -f records/reconstruction.completed; do
  if (( SECONDS > deadline )); then echo 'Timed out waiting for reconstruction'; exit 1; fi
  sleep 10
done
envs/reconstruction/bin/python scripts/render_common.py --ply outputs/reconstruction/point_cloud/iteration_30000/point_cloud.ply --name reconstruction --depth
envs/reconstruction/bin/python scripts/render_target_masks.py
date -Iseconds > records/baseline.rendering.completed
