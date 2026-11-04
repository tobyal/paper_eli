export ROOT=/mnt/sda/ubuntu/lkx/papers/paper_eli
export http_proxy=http://127.0.0.1:17891
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export HF_HOME="$ROOT/models/huggingface" TORCH_HOME="$ROOT/models/torch"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
