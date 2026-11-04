set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GSLE_GPU:-0}"
until envs/download/bin/python -c 'import json; assert "openclip" in json.load(open("records/gsle/models.json"))' 2>/dev/null; do sleep 20; done
envs/gsle_langsplat/bin/python gsle/prepare_dataset.py
export GSLE_DATASET_PREPARED=1
GSLE_GPU=0 bash gsle/03_features.sh --shard 0 --shards 3 > logs/gsle/03_features_shard0.log 2>&1 &
pid0=$!
GSLE_GPU=2 bash gsle/03_features.sh --shard 1 --shards 3 > logs/gsle/03_features_shard1.log 2>&1 &
pid1=$!
GSLE_GPU=3 bash gsle/03_features.sh --shard 2 --shards 3 > logs/gsle/03_features_shard2.log 2>&1 &
pid2=$!
wait "$pid0"
wait "$pid1"
wait "$pid2"
envs/gsle_langsplat/bin/python -u gsle/train_autoencoder.py > logs/gsle/04_autoencoder.log 2>&1
cd "$ROOT/repos/LangSplat"
jobs=()
for pair in 1:0 2:2 3:3; do
  level="${pair%:*}"
  gpu="${pair#*:}"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/envs/gsle_langsplat/bin/python" -u train.py -s "$ROOT/data/gsle_langsplat" -m "$ROOT/outputs/gsle/langsplat" -r 2 --data_device cpu --feature_level "$level" --start_checkpoint "$ROOT/outputs/reconstruction/chkpnt30000.pth" --port "$((55650+level))" > "$ROOT/logs/gsle/04_language_level${level}.log" 2>&1 &
  jobs+=("$!")
done
for pid in "${jobs[@]}"; do wait "$pid"; done
cd "$ROOT"
envs/gsle_langsplat/bin/python gsle/validate_trained_language.py --iteration 30000
date -Is > "$ROOT/records/gsle/language.completed"
