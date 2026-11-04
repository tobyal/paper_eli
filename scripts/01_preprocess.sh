set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
mkdir -p runtime/colmap-lib data/industrial/input data/industrial/distorted/sparse
ln -sfn "$ROOT/runtime/glog04/lib/libglog.so.0" runtime/colmap-lib/libglog.so.0
ln -sfn /home/ubuntu/anaconda3/lib/libgflags.so.2.2.2 runtime/colmap-lib/libgflags.so.2.2
export LD_LIBRARY_PATH="$ROOT/runtime/colmap-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM=offscreen
colmap -h > records/colmap_version.txt
ffprobe -v error -show_format -show_streams -of json data/raw/20261016-142321.mp4 > records/video_metadata.json
sha256sum data/raw/20261016-142321.mp4 > records/video_sha256.txt
if ! test -f data/industrial/input/.complete; then
  ffmpeg -hide_banner -loglevel warning -i data/raw/20261016-142321.mp4 -vf 'fps=2,scale=1920:1080:flags=lanczos' -q:v 2 data/industrial/input/frame_%06d.jpg
  touch data/industrial/input/.complete
fi
colmap feature_extractor --database_path data/industrial/distorted/database.db --image_path data/industrial/input --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0 --SiftExtraction.num_threads 8
colmap sequential_matcher --database_path data/industrial/distorted/database.db --SiftMatching.use_gpu 1 --SiftMatching.gpu_index 0 --SequentialMatching.overlap 20 --SequentialMatching.quadratic_overlap 1 --SequentialMatching.loop_detection 0
colmap mapper --database_path data/industrial/distorted/database.db --image_path data/industrial/input --output_path data/industrial/distorted/sparse --Mapper.num_threads 12 --Mapper.ba_global_function_tolerance 0.000001
for model in data/industrial/distorted/sparse/[0-9]*; do
  colmap model_analyzer --path "$model" > "records/colmap_model_$(basename "$model").txt" 2>&1
done
# Select the reconstruction with the greatest number of registered images.
best_model=$(python3 - <<'PY'
import pathlib,struct
models=list(pathlib.Path('data/industrial/distorted/sparse').glob('[0-9]*'))
print(max(models,key=lambda p:struct.unpack('<Q',(p/'images.bin').read_bytes()[:8])[0]))
PY
)
colmap image_undistorter --image_path data/industrial/input --input_path "$best_model" --output_path data/industrial --output_type COLMAP --max_image_size 1920
mkdir -p data/industrial/sparse/0
for file in cameras.bin images.bin points3D.bin; do
  if test -f "data/industrial/sparse/$file"; then mv "data/industrial/sparse/$file" "data/industrial/sparse/0/$file"; fi
done
printf '%s\n' "$best_model" > records/selected_colmap_model.txt
date -Iseconds > records/preprocess.completed
