#!/usr/bin/env bash
# Project-local dependencies; no system installs or other users' environments.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TOOLS="$ROOT/industrial_swap_v2/tools"
mkdir -p "$TOOLS/downloads"
NAME=blender-4.5.4-linux-x64
if [[ ! -x "$TOOLS/$NAME/blender" ]]; then
  curl -fL --retry 2 --proxy http://127.0.0.1:17891 \
    "https://download.blender.org/release/Blender4.5/$NAME.tar.xz" -o "$TOOLS/downloads/$NAME.tar.xz"
  printf '%s  %s\n' 2e6ef8e99fc36327270429ddc8e7bad2859dd878a5a137d2e0bf0f02f6792505 \
    "$TOOLS/downloads/$NAME.tar.xz" | sha256sum -c -
  tar -xJf "$TOOLS/downloads/$NAME.tar.xz" -C "$TOOLS"
fi
"$ROOT/envs/gsle/bin/pip" install --no-deps --proxy http://127.0.0.1:17891 \
  trimesh==4.6.8 rtree==1.4.1 OpenEXR==3.3.2
"$TOOLS/$NAME/blender" --version
