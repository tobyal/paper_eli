#!/usr/bin/env bash
# All model inputs are reused; this script never calls Qwen or TRELLIS.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
config="${1:-industrial_swap_v2/webapp/data/experiments/9b0eeee5840d453e8f21e1763544a6cb/runs/173dd1e87749494db5f66ab1287d37d7/config.yaml}"
run="${2:-reproduce_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo 'Invalid run name' >&2; exit 2; fi
export SWAP_GPU="${SWAP_GPU:-1}"
export CUDA_VISIBLE_DEVICES="$SWAP_GPU" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
available="$(nvidia-smi -i "$SWAP_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( available < 16000 )); then echo 'Need at least 16 GB free on the selected shared GPU; select another available GPU.' >&2; exit 3; fi
base="outputs/background_repair/$run"
logs="industrial_swap_v2/logs/background_repair/$run"
mkdir -p "$logs"
py=envs/gsle/bin/python
"$py" -m industrial_swap_v2.inspect_background_repair --config "$config" --output "$base/inspection" 2>&1 | tee "$logs/inspection.log"
"$py" -m industrial_swap_v2.repair_background --config "$config" --inspection "$base/inspection" --output "$base/fixed" 2>&1 | tee "$logs/repair.log"
"$py" -m industrial_swap_v2.export_background_repair --config "$config" --repair "$base/fixed" 2>&1 | tee "$logs/export.log"
printf 'Repaired scene: %s/industrial_swap_v2/%s/fixed/scene_coarse_fixed.ply\n' "$root" "$base"
