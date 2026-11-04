set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/runtime/colmap-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM=offscreen
best_model=$(python3 - <<'PY'
from pathlib import Path
import struct
models=list(Path('data/industrial/distorted/sparse').glob('[0-9]*'))+list(Path('data/industrial/distorted/sparse_exhaustive').glob('[0-9]*'))
if Path('data/industrial/distorted/sparse_exhaustive/images.bin').exists():
    models.append(Path('data/industrial/distorted/sparse_exhaustive'))
def count(p):
    with (p/'images.bin').open('rb') as f: return struct.unpack('<Q',f.read(8))[0]
best=max(models,key=count)
assert count(best)>=315, f'Only {count(best)}/350 frames registered; inspect before training'
print(best)
PY
)
colmap model_analyzer --path "$best_model" > records/final_sfm_quality.txt 2>&1
python3 - <<'PY'
import re
from pathlib import Path
text=Path('records/final_sfm_quality.txt').read_text()
error=float(re.search(r'Mean reprojection error: ([0-9.]+)px',text).group(1))
assert error<1.5, f'Reprojection error {error}px needs review'
PY
colmap image_undistorter --image_path data/industrial/input --input_path "$best_model" --output_path data/industrial --output_type COLMAP --max_image_size 1920
mkdir -p data/industrial/sparse/0
for file in cameras.bin images.bin points3D.bin; do
  if test -f "data/industrial/sparse/$file"; then mv -f "data/industrial/sparse/$file" "data/industrial/sparse/0/$file"; fi
done
printf '%s\n' "$best_model" > records/selected_colmap_model.txt
date -Iseconds > records/preprocess.verified
cat records/final_sfm_quality.txt
