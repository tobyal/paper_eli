set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
# Torch is shared read-only from the GE environment; CUDA extensions are isolated.
deadline=$((SECONDS + 7200))
until "$ROOT/envs/gaussianeditor/bin/python" -c 'import torch, torchvision, numpy; assert torch.cuda.is_available()' >/dev/null 2>&1 && test -f records/preprocess.verified && test -f records/reconstruction_commit.txt; do
  if (( SECONDS > deadline )); then echo 'Timed out waiting for preprocessing and base torch environment'; exit 1; fi
  sleep 30
done
test -f records/reconstruction.environment.completed
export PATH="$ROOT/envs/reconstruction/bin:$PATH"
cd repos/gaussian-splatting
/usr/bin/time -v -o "$ROOT/logs/05_reconstruction.time.txt" python train.py -s "$ROOT/data/industrial" -m "$ROOT/outputs/reconstruction" --eval -r 2 --iterations 30000 --save_iterations 7000 30000 --test_iterations 7000 30000 --checkpoint_iterations 7000 30000 --ip 127.0.0.1 --port 16909
python render.py -m "$ROOT/outputs/reconstruction" --iteration 30000
python "$ROOT/scripts/prepare_edit_dataset.py"
date -Iseconds > "$ROOT/records/reconstruction.completed"
