set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
deadline=$((SECONDS + 10800))
until test -f records/gaussianeditor.environment.completed; do
  if (( SECONDS > deadline )); then echo 'Timed out waiting for verified GaussianEditor environment'; exit 1; fi
  sleep 20
done
bash scripts/06_environment_dge.sh > logs/06_environment_dge.log 2>&1 &
dge_pid=$!
bash scripts/07_environment_dreamcatalyst.sh > logs/07_environment_dreamcatalyst.log 2>&1 &
dc_pid=$!
wait "$dge_pid"
wait "$dc_pid"
