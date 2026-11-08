#!/usr/bin/env bash
# Query/localize/render only. Never submits Qwen or runs M3/M4.
set -euo pipefail
ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli
cd "$ROOT"
requested_batch=${1:?Provide a NEW absolute batch output path}
batch_path=$(realpath -m -- "$requested_batch")
case "$batch_path/" in
  "$ROOT/industrial_swap_v2/outputs/case_preparation/"*) ;;
  *) echo 'Batch must remain in industrial_swap_v2/outputs/case_preparation' >&2; exit 2 ;;
esac
if [[ -e "$batch_path" ]]; then echo 'Output exists; refusing to overwrite' >&2; exit 2; fi
mkdir -p -- "$batch_path"
export CUDA_VISIBLE_DEVICES=${SWAP_GPU:-2}
export SWAP_DISABLE_QWEN=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
for round in 1 2 3 4 5; do
  suffix=$round
  queries="industrial_swap_v2/configs/case_queries_round${round}.json"
  if [[ "$round" == 1 ]]; then suffix=''; queries=industrial_swap_v2/configs/case_queries.json; fi
  envs/gsle_langsplat/bin/python -u -m industrial_swap_v2.scripts.survey_case_semantics \
    --output "$batch_path/survey$suffix" --queries "$queries" > "$batch_path/survey${round}.log" 2>&1
  envs/gsle/bin/python -u -m industrial_swap_v2.scripts.localize_case_candidates \
    --batch "$batch_path" --survey "survey$suffix" --candidates "candidates$suffix" > "$batch_path/localization${round}.log" 2>&1
done
envs/gsle/bin/python -u -m industrial_swap_v2.scripts.prepare_case_inputs \
  --batch "$batch_path" --selection industrial_swap_v2/configs/cases_10_selection.json > "$batch_path/prepare.log" 2>&1
envs/gsle/bin/python -m unittest industrial_swap_v2.tests.test_case_preparation industrial_swap_v2.tests.test_qwen_once -v > "$batch_path/interface_tests.log" 2>&1
envs/gsle/bin/python -u -m industrial_swap_v2.scripts.audit_case_inputs \
  --batch "$batch_path" > "$batch_path/integrity_audit.log" 2>&1
echo "Prepared and stopped before Qwen: $batch_path"
echo 'Human inspection is required. No generated image or 3D replacement was produced.'
