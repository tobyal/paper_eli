#!/usr/bin/env bash
set -euo pipefail
V2=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=$(dirname "$V2")
cd "$PROJECT"
source scripts/common.sh
exec envs/gsle/bin/python industrial_swap_v2/run.py --stage all "$@"
