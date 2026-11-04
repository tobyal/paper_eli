"""Frozen Stable Zero123 3D-SDS and GSLE depth/inpainting SDS (Eq5-8)."""

import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, config


def model_paths():
    return json.loads((ROOT / "records/gsle/models.json").read_text())


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


class StableZeroGuidance:
    def __init__(self, reference_rgb, reference_elevation):
        self.cfg = config()
        sys.path.insert(0, str(ROOT / "repos/DreamGaussian"))
        from guidance.zero123_utils import Zero123

        self.guide = Zero123(
            "cuda",
            fp16=True,
            t_range=self.cfg["diffusion_timesteps"],
            model_key=model_paths()["stable_zero123"]["path"],
        )
        self.guide.use_stable_zero123 = True
        for module in (
            self.guide.pipe.vae,
            self.guide.pipe.unet,
            self.guide.pipe.image_encoder,
            self.guide.pipe.clip_camera_projection,
        ):
            freeze(module)
        self.guide.get_img_embeds(reference_rgb[None])
        # DreamGaussian's orbit elevation is positive downward, opposite our +up.
        self.reference_elevation = float(reference_elevation)

    def loss(self, rgb, elevation, azimuth):
        return self.loss_batch(rgb[None], [elevation], [azimuth])

    def loss_batch(self, rgb, elevations, azimuths):
        return self.guide.train_step(
            rgb,
            [-(e - self.reference_elevation) for e in elevations],
            azimuths,
            [0.0] * len(elevations),
            guidance_scale=self.cfg["coarse"]["guidance_scale"],
            default_elevation=-self.reference_elevation,
        )


class DepthInpaintingGuidance:
    def __init__(self):
        self.cfg = config()
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
            DDIMScheduler,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        paths = model_paths()
        dtype = torch.float16
        control = ControlNetModel.from_pretrained(
            paths["controlnet_depth"]["path"],
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=False,
            local_files_only=True,
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            paths["inpainting"]["path"],
            controlnet=control,
            torch_dtype=dtype,
            variant="fp16",
            local_files_only=True,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler
        # Paper Sec5.1 uses SD1.5 for view-dependent text embeddings.
        self.view_tokenizer = CLIPTokenizer.from_pretrained(
            paths["view_text"]["path"], subfolder="tokenizer", local_files_only=True
        )
        self.view_text = freeze(
            CLIPTextModel.from_pretrained(
                paths["view_text"]["path"],
                subfolder="text_encoder",
                torch_dtype=dtype,
                local_files_only=True,
            ).cuda()
        )
        for module in (
            self.pipe.vae,
            self.pipe.unet,
            self.pipe.text_encoder,
            self.pipe.controlnet,
        ):
            freeze(module)
        assert (
            self.pipe.unet.config.in_channels == 9
        ), "DI-SDS requires the inpainting UNet, not a 4-channel text-to-image UNet"
        assert self.pipe.controlnet.config.in_channels == 4
        self.vae = self.pipe.vae
        self.dtype = dtype
        self.alphas = self.scheduler.alphas_cumprod.cuda()
        self.min_step = int(1000 * self.cfg["diffusion_timesteps"][0])
        self.max_step = int(1000 * self.cfg["diffusion_timesteps"][1])
        self.text_cache = {}

    @torch.no_grad()
    def text_embeddings(self, prompt):
        if prompt not in self.text_cache:
            tokens = self.view_tokenizer(
                [prompt, self.cfg["negative_text"]],
                padding="max_length",
                max_length=self.view_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.text_cache[prompt] = self.view_text(tokens.input_ids.cuda())[0]
        return self.text_cache[prompt]

    @torch.no_grad()
    def inpaint(self, image, mask, depth, prompt, seed):
        cfg = self.cfg["inpainting"]
        return self.pipe(
            prompt=prompt,
            negative_prompt=cfg.get("negative_text", self.cfg["negative_text"]),
            image=image,
            mask_image=mask,
            control_image=depth,
            height=512,
            width=512,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"],
            controlnet_conditioning_scale=cfg["controlnet_scale"],
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    def loss(self, rgb, depth, mask, prompt):
        loss, diagnostics = self.loss_batch(
            rgb[None], depth[None], mask[None], [prompt]
        )
        return loss, diagnostics[0]

    def loss_batch(self, rgb, depth, mask, prompts):
        """Only VAE encoder/render path tracks gradients; all teachers stay frozen."""
        batch = len(rgb)
        image = rgb.to(self.dtype) * 2 - 1
        latents = (
            self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        )
        with torch.no_grad():
            m = mask.to(self.dtype)
            masked_image = image.detach() * (m < 0.5)
            masked_latents = (
                self.vae.encode(masked_image).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            latent_mask = F.interpolate(m, size=latents.shape[-2:], mode="nearest")
            timestep = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch,),
                device="cuda",
                dtype=torch.long,
            )
            noise = torch.randn_like(latents)
            noisy = self.scheduler.add_noise(latents.detach(), noise, timestep)
            noisy_pair = torch.cat((noisy, noisy))
            time_pair = timestep.repeat(2)
            embeddings = [self.text_embeddings(prompt) for prompt in prompts]
            texts = torch.cat(
                [item[:1] for item in embeddings] + [item[1:] for item in embeddings]
            )
            control_image = depth.detach().to(self.dtype)
            if self.cfg["refine"]["unconditional_conditioning"] == "drop_all_eq7":
                down, mid = self.pipe.controlnet(
                    noisy,
                    timestep,
                    encoder_hidden_states=texts[:batch],
                    controlnet_cond=control_image,
                    conditioning_scale=1.0,
                    return_dict=False,
                )
                down = [torch.cat((value, torch.zeros_like(value))) for value in down]
                mid = torch.cat((mid, torch.zeros_like(mid)))
                mask_pair = torch.cat((latent_mask, torch.zeros_like(latent_mask)))
                context_pair = torch.cat(
                    (masked_latents, torch.zeros_like(masked_latents))
                )
            else:
                # Explicit alternative for ablation: conventional text-only CFG.
                down, mid = self.pipe.controlnet(
                    noisy_pair,
                    time_pair,
                    encoder_hidden_states=texts,
                    controlnet_cond=control_image.repeat(2, 1, 1, 1),
                    conditioning_scale=1.0,
                    return_dict=False,
                )
                mask_pair = latent_mask.repeat(2, 1, 1, 1)
                context_pair = masked_latents.repeat(2, 1, 1, 1)
            nine = torch.cat((noisy_pair, mask_pair, context_pair), dim=1)
            prediction = self.pipe.unet(
                nine,
                time_pair,
                encoder_hidden_states=texts,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
            ).sample
            conditioned, unconditioned = prediction.chunk(2)
            eps = conditioned + self.cfg["refine"]["guidance_scale"] * (
                conditioned - unconditioned
            )
            gradient = (1 - self.alphas[timestep]).reshape(batch, 1, 1, 1) * (
                eps.float() - noise.float()
            )
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite DI-SDS gradient")
        target = (latents.float() - gradient).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction="sum")
        diagnostics = [
            {
                "timestep": int(timestep[i]),
                "noise_residual_rms": float(gradient[i].square().mean().sqrt()),
                "latent_shape": list(latents[i : i + 1].shape),
                "inpainting_channels": int(nine.shape[1]),
                "controlnet_channels": int(noisy_pair.shape[1]),
                "unconditional_conditioning": self.cfg["refine"][
                    "unconditional_conditioning"
                ],
            }
            for i in range(batch)
        ]
        return loss, diagnostics
