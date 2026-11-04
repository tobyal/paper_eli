set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/runtime/colmap-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM=offscreen
mkdir -p data/industrial/distorted/sparse_exhaustive
colmap exhaustive_matcher --database_path data/industrial/distorted/database.db --SiftMatching.use_gpu 1 --SiftMatching.gpu_index 0 --SiftMatching.num_threads 8
colmap mapper --database_path data/industrial/distorted/database.db --image_path data/industrial/input --input_path data/industrial/distorted/sparse/0 --output_path data/industrial/distorted/sparse_exhaustive --Mapper.num_threads 12 --Mapper.ba_global_function_tolerance 0.000001
if test -f data/industrial/distorted/sparse_exhaustive/images.bin; then
  colmap model_analyzer --path data/industrial/distorted/sparse_exhaustive > records/colmap_exhaustive_0.txt 2>&1
else
  for model in data/industrial/distorted/sparse_exhaustive/[0-9]*; do
    test -f "$model/images.bin" || continue
    colmap model_analyzer --path "$model" > "records/colmap_exhaustive_$(basename "$model").txt" 2>&1
  done
fi
