set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
# Run only after visually inspecting the actual five-view RGB sheet.
envs/download/bin/python scripts/record_method_review.py editsplat \
  --label execution_success_edit_target_not_achieved \
  --result 'The target housing remains predominantly white in all five inspected views. Some support edges become rougher and nearby surfaces acquire slight color changes. No convincing red housing edit is achieved.' \
  --result-zh '五个检查视角中的目标外壳仍主要为白色，没有明显改红；部分支架边缘更粗糙，附近表面出现轻微颜色变化。'
nohup envs/editsplat/bin/python scripts/evaluate_common.py > logs/15_evaluate_all.log 2>&1 < /dev/null &
echo "Evaluation PID: $!"
nohup envs/reconstruction/bin/python scripts/make_comparison.py > logs/15b_make_comparison.log 2>&1 < /dev/null &
echo "Comparison PID: $!"
envs/download/bin/python scripts/record_checkpoint_artifacts.py > logs/15c_checkpoint_hashes.log
