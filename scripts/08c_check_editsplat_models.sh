set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
deadline=$((SECONDS + 7200))
until test -f records/editsplat.environment.completed; do
  if (( SECONDS > deadline )); then echo 'Timed out waiting for EditSplat environment'; exit 1; fi
  sleep 10
done
envs/editsplat/bin/python scripts/08c_check_editsplat_models.py
