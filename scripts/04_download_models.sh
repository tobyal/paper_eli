set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
if ! test -x envs/download/bin/python; then
  /mnt/sda/ubuntu/anaconda3/envs/lkx_spags/bin/python -m venv envs/download
fi
envs/download/bin/python -m pip install huggingface_hub==0.25.2
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
for group in core aux editsplat_safety imagereward bert_safe; do
  envs/download/bin/python scripts/download_hf_parallel.py "$group"
done
envs/download/bin/python scripts/download_aux_models.py
envs/download/bin/python scripts/04b_lpips.py
envs/download/bin/python scripts/04c_clip.py
