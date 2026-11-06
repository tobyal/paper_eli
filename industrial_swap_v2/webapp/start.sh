#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.manage start --port "${SWAP_WEB_PORT:-18086}"
