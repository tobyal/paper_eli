set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
if ! test -d repos/gaussian-splatting/.git; then
  git clone --filter=blob:none https://github.com/graphdeco-inria/gaussian-splatting.git repos/gaussian-splatting
fi
cd repos/gaussian-splatting
# Use the original 2023 implementation, before later depth/exposure extensions.
commit=2eee0e26d2d5fd00ec462df47752223952f6bf4e
git checkout "$commit"
git submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn
git rev-parse HEAD > "$ROOT/records/reconstruction_commit.txt"
git submodule status --recursive > "$ROOT/records/reconstruction_submodules.txt"
