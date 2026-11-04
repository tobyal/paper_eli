set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
until envs/download/bin/python -c 'import json; assert "openclip" in json.load(open("records/gsle/models.json"))' 2>/dev/null; do sleep 20; done
bash gsle/03_features.sh
cd "$ROOT/repos/LangSplat/autoencoder"
"$ROOT/envs/gsle_langsplat/bin/python" -u train.py --dataset_path "$ROOT/data/gsle_langsplat" --dataset_name industrial_gsle --num_epochs 100 --lr 0.0007 > "$ROOT/logs/gsle/04_autoencoder.log" 2>&1
"$ROOT/envs/gsle_langsplat/bin/python" -u test.py --dataset_path "$ROOT/data/gsle_langsplat" --dataset_name industrial_gsle > "$ROOT/logs/gsle/04_encode.log" 2>&1
cd "$ROOT/repos/LangSplat"
for level in 1 2 3; do
  "$ROOT/envs/gsle_langsplat/bin/python" -u train.py -s "$ROOT/data/gsle_langsplat" -m "$ROOT/outputs/gsle/langsplat" -r 2 --data_device cpu --feature_level "$level" --start_checkpoint "$ROOT/outputs/reconstruction/chkpnt30000.pth" --port "$((55650+level))" > "$ROOT/logs/gsle/04_language_level${level}.log" 2>&1
done
cd "$ROOT"
envs/gsle_langsplat/bin/python gsle/validate_trained_language.py --iteration 30000
date -Is > "$ROOT/records/gsle/language.completed"
