#!/usr/bin/env bash
# Reuse an existing M3. Never runs Qwen, TRELLIS, alignment, or floor repair.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
config="${1:?Pass the existing M3 config.yaml}"
run="${2:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run" =~ ^[a-zA-Z0-9_-]+$ ]]; then echo 'Invalid run name' >&2; exit 2; fi
export SWAP_GPU="${SWAP_GPU:-1}" CUDA_VISIBLE_DEVICES="${SWAP_GPU:-1}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
free="$(nvidia-smi -i "$SWAP_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( free < 10240 )); then echo 'Select a shared GPU with at least 10 GiB free.' >&2; exit 3; fi
output="outputs/oagof/$run"; logs="industrial_swap_v2/logs/oagof/$run"
mkdir -p "$logs"
options=()
if [[ -n "${3:-}" ]]; then options+=(--settings "$3"); fi
envs/gsle/bin/python -m industrial_swap_v2.modules.m3_occlusion_filter \
  --config "$config" --output "$output" "${options[@]}" 2>&1 | tee "$logs/filter.log"
envs/gsle/bin/python -m industrial_swap_v2.scripts.validate_oagof \
  --config "$config" --filter-output "$output" 2>&1 | tee "$logs/validation.log"
printf 'Verified scene: %s/industrial_swap_v2/%s/scene_filtered.ply\n' "$root" "$output"
