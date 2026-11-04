set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
until test -f outputs/reconstruction/chkpnt7000.pth; do sleep 10; done
# Checkpoint saving follows PLY saving in the untouched official entrypoint.
sleep 15
envs/reconstruction/bin/python scripts/render_common.py --ply outputs/reconstruction/point_cloud/iteration_7000/point_cloud.ply --name reconstruction_7000_preview --frames 1,51,111,151,191,241,291,350
