set -euo pipefail
ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli
mkdir -p "$ROOT"/{data/raw,data/industrial,repos,envs,models,outputs,logs,records,scripts}
export HF_HOME="$ROOT/models/huggingface"
export TORCH_HOME="$ROOT/models/torch"
export GIT_TERMINAL_PROMPT=0
for entry in buaacyw/GaussianEditor silent-chen/DGE kaist-cvml/DreamCatalyst kuai-lab/cvpr25_EditSplat; do
  name=${entry##*/}
  case "$name" in
    GaussianEditor) revision=9249f847c57036266a0bfb3a210681f94264ede9 ;;
    DGE) revision=7eed0bc57824707c53b935bd19ea994f19f41eae ;;
    DreamCatalyst) revision=072cb248ce847cef0e443d70baef7db8cee89ee7 ;;
    cvpr25_EditSplat) revision=631f26679f9d66c7099413be441d346185c6442f ;;
  esac
  if ! test -d "$ROOT/repos/$name/.git"; then
    git init "$ROOT/repos/$name"
    git -C "$ROOT/repos/$name" remote add origin "https://github.com/$entry.git"
    git -C "$ROOT/repos/$name" fetch --depth 1 origin "$revision"
    git -C "$ROOT/repos/$name" checkout --detach FETCH_HEAD
    git -C "$ROOT/repos/$name" submodule update --init --recursive --depth 1
  fi
  test "$(git -C "$ROOT/repos/$name" rev-parse HEAD)" = "$revision"
  git -C "$ROOT/repos/$name" rev-parse HEAD
  git -C "$ROOT/repos/$name" submodule status --recursive || true
done
