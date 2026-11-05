#!/usr/bin/env bash
set -euo pipefail
V2=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=$(dirname "$V2")
# User-supplied working environment, read-only system packages. New packages,
# if required, must be installed only into this new virtualenv.
BASE=/mnt/sda/ubuntu/anaconda3/envs/trellis2
"$BASE/bin/python" -m venv --system-site-packages "$V2/envs/trellis2_reused"
cd "$PROJECT"
envs/gsle/bin/python -m industrial_swap_v2.scripts.reuse_existing_models
"$V2/envs/trellis2_reused/bin/python" -m pip freeze > "$V2/records/trellis2_reused.pip.freeze.txt"
"$V2/envs/trellis2_reused/bin/python" -m industrial_swap_v2.scripts.record_environment
