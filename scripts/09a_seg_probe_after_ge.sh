set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
deadline=$((SECONDS + 7200))
until test -f records/gaussianeditor.environment.completed; do
  if (( SECONDS > deadline )); then echo 'Timed out waiting for verified GE environment'; exit 1; fi
  sleep 10
done
envs/gaussianeditor/bin/python scripts/probe_segmentation.py
