set -euo pipefail
source /mnt/sda/ubuntu/lkx/papers/paper_eli/scripts/common.sh
cd "$ROOT"
# Symbol inspection identifies glog 0.4 (kLogSiteUninitialized), removed in 0.5.
# Build a private ABI-matching library instead of changing any system libraries.
if ! test -f runtime/glog04/lib/libglog.so.0; then
  curl -fL --retry 3 https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz -o runtime/glog-0.4.0.tar.gz
  tar -xzf runtime/glog-0.4.0.tar.gz -C runtime
  cmake -S runtime/glog-0.4.0 -B runtime/glog04-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/runtime/glog04" -DBUILD_TESTING=OFF -DWITH_GFLAGS=OFF -DBUILD_SHARED_LIBS=ON
  cmake --build runtime/glog04-build -j8
  cmake --install runtime/glog04-build
fi
ln -sfn "$ROOT/runtime/glog04/lib/libglog.so.0" runtime/colmap-lib/libglog.so.0
LD_LIBRARY_PATH="$ROOT/runtime/colmap-lib" colmap -h > records/colmap_version.txt
