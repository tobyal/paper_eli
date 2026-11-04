set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
envs/reconstruction/bin/python -m pip install --no-deps repos/dependencies/segment-anything opencv-python==4.8.1.78
envs/reconstruction/bin/python scripts/probe_reference_sam.py
