"""Resolve required pretrained assets once, then reuse immutable revisions."""

import json, os
from pathlib import Path
from huggingface_hub import HfApi

root = Path(os.environ["ROOT"])
dest = root / "records/gsle/model_pins.json"
pins = json.loads(dest.read_text()) if dest.exists() else {}
models = {
    "view_text": (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ["tokenizer/*", "text_encoder/config.json", "text_encoder/pytorch_model.bin"],
    ),
    "openclip": (
        "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        ["open_clip_pytorch_model.bin", "open_clip_config.json"],
    ),
    "inpainting": (
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "feature_extractor/*",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
        ],
    ),
    "controlnet_depth": (
        "lllyasviel/control_v11f1p_sd15_depth",
        ["config.json", "diffusion_pytorch_model.fp16.bin"],
    ),
    "stable_zero123": (
        "ashawkey/stable-zero123-diffusers",
        [
            "model_index.json",
            "scheduler/*",
            "feature_extractor/*",
            "image_encoder/config.json",
            "image_encoder/model.safetensors",
            "clip_camera_projection/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    ),
}
for name, (repo, patterns) in models.items():
    if name not in pins:
        info = HfApi().model_info(
            repo,
            revision=(
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
                if name == "view_text"
                else None
            ),
            files_metadata=True,
        )
        pins[name] = {
            "repository": repo,
            "revision": info.sha,
            "patterns": patterns,
            "available_files": [
                {"name": f.rfilename, "bytes": f.size} for f in info.siblings
            ],
        }
        dest.write_text(json.dumps(pins, indent=2))
    # File patterns may be corrected after inspecting the immutable file manifest.
    pins[name]["patterns"] = patterns
    dest.write_text(json.dumps(pins, indent=2))
    print(name, pins[name]["revision"], flush=True)
