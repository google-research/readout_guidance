# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bitsandbytes as bnb
import gc
from IPython.display import display
from IPython.display import clear_output
import numpy as np
from omegaconf import ListConfig
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from diffusers import DDIMScheduler
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_xl_adapter import _preprocess_adapter_image

from dhf.stable_diffusion.resnet import init_resnet_func, collect_feats, collect_channels
from readout_guidance import rg_operators, rg_helpers

class ReadoutGuidance():
    def __init__(
        self,
        model=None,
        edits=[],
        points=None,
        latent_dim=None,
        save_mode="hidden",
        idxs=None
    ):
        if model is not None:
            self.model = model
            if edits:
                init_resnet_func(self.model.unet, save_mode=save_mode, idxs=idxs, reset=True)
            self.channels = collect_channels(self.model.unet)
        else:
            self.model = None

        self.idxs = idxs
        self.latent_dim = latent_dim
        self.edits = edits
        self.points = points
        self.gt_feat = None
        self.obs_feat = None
    
    def collect_and_resize_feats(self):
        return rg_helpers.collect_and_resize_feats(self.model, self.idxs, self.latent_dim)

    def has_edits(self):
        return len(self.edits) > 0
    
def diffusion_step(
        model,
        controller, 
        latents, 
        context,
        added_cond_kwargs,
        i, 
        t,
        next_t,
        guidance_scale,
        log=False,
        scheduler_kwargs={},
        num_optimizer_steps=[], 
        lr=2e-2,
        low_memory=True,
        pbar=None
    ):
    
    def prepare_control(guess_mode=False):
        control_image = [edit["control_image"] for edit in controller.edits if "control_image" in edit][0]
        control_image_input = model.prepare_image(
            control_image, 
            control_image.size[0], 
            control_image.size[1], 
            latents.shape[0], 
            1, 
            latents.device, 
            latents.dtype, 
            True, 
            guess_mode
        )
        # Infer ControlNet only for the conditional batch.
        control_model_input = torch.cat([latents] * 2)
        controlnet_prompt_embeds = context
        down_block_res_samples, mid_block_res_sample = model.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image_input,
            conditioning_scale=1.0,
            guess_mode=guess_mode,
            return_dict=False,
        )
        return {
            "down_block_additional_residuals": down_block_res_samples,
            "mid_block_additional_residual": mid_block_res_sample
        }
    
    def prepare_adapter():
        control_image = [edit["control_image"] for edit in controller.edits if "control_image" in edit][0]
        adapter_input = _preprocess_adapter_image(control_image, control_image.size[0], control_image.size[1])
        adapter_input = adapter_input.to(model.adapter.device).to(model.adapter.dtype)
        adapter_state = model.adapter(adapter_input)
        adapter_conditioning_scale = 1.0
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * adapter_conditioning_scale
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2 * latents.shape[0], dim=0)
        down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
        return {
            "down_block_additional_residuals": down_intrablock_additional_residuals
        }

    def _diffusion_step(latents):
        unet_kwargs = {}
        unet_context = context
        if hasattr(model, "controlnet"):
            unet_kwargs = prepare_control()
        elif hasattr(model, "adapter"):
            unet_kwargs = prepare_adapter()
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(
            latents_input,
            t,
            encoder_hidden_states=unet_context,
            added_cond_kwargs=added_cond_kwargs,
            **unet_kwargs
        )["sample"]
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        if type(guidance_scale) is ListConfig or type(guidance_scale) is list:
            guidance_scale_pt = torch.tensor(guidance_scale)[:, None, None, None]
            guidance_scale_pt = guidance_scale_pt.to(noise_pred.device).to(noise_pred.dtype)
        else:
            guidance_scale_pt = guidance_scale
        noise_pred = noise_pred_uncond + guidance_scale_pt * (noise_pred_cond - noise_pred_uncond)
        if controller.edits:
            feats = controller.collect_and_resize_feats()
        else:
            feats = None
        return noise_pred, feats
    
    def optimize_latents(latents):
        # Use gradient checkpointing which recomputes
        # intermediates on the backwards pass to save memory.
        latents.requires_grad_(True)
        torch.set_grad_enabled(True)
        optimizer_cls = bnb.optim.AdamW8bit if low_memory else torch.optim.AdamW
        optimizer = optimizer_cls([latents], lr=lr, weight_decay=0.0)
        losses = []
        for j in range(num_optimizer_steps[i]):
            loss = 0
            noise_pred, feats = checkpoint(_diffusion_step, latents)
            emb = rg_helpers.embed_timestep(model.unet, latents, t)
            b = feats.shape[0]
            # Compute the loss over both the uncond and cond branch
            for gt_idx in [0, b//2]:
                log_branch = (log and gt_idx != 0)
                batch_idx = gt_idx + 1
                latents_scale = (latents.detach().min(), latents.detach().max())
                rg_loss = rg_operators.loss_guidance(controller, feats, batch_idx, gt_idx, edits=controller.edits, log=log_branch, emb=emb, latents_scale=latents_scale, t=t, i=i)
                loss += rg_loss
            losses.append(loss.item())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                optimizer.step()
        return latents.detach(), losses
    # Clear previous logged outputs
    if log:
        clear_output()
    if controller.has_edits(): 
        latents, losses = optimize_latents(latents)
        if losses:
            rg_loss = np.mean(losses)
        else:
            rg_loss = "N/A"
    else:
        rg_loss = "N/A"
    # Perform a normal diffusion step
    noise_pred, feats = checkpoint(_diffusion_step, latents)
    # rg_helpers scheduler update is needed to do DDIM inversion
    latents, latents_x0s = rg_helpers.get_xt_next(
        model.scheduler,
        noise_pred,
        t, next_t,
        latents,
        eta=scheduler_kwargs["eta"], 
        variance_noise=scheduler_kwargs["zs"][i]
    )
    if pbar is not None:
        description = f"Readout Guidance Loss {rg_loss}"
        pbar.set_description(description)
    if log:
        with torch.no_grad():
            image = rg_helpers.decode_latents(model.vae, latents_x0s)
            image = rg_helpers.view_images(image)
            display(image)
        if controller.gt_feat is not None and controller.obs_feat is not None:
            image = rg_helpers.view_images([controller.gt_feat, controller.obs_feat])
            display(image)
    torch.cuda.empty_cache()
    gc.collect()
    return latents

def text2image_rg(
    model,
    controller,
    prompt,
    latents,
    text_weight: float = 7.5,
    rg_weight: float = 2e-2,
    rg_ratio: list = [0.0, 1.0],
    num_recurrent_steps: int = 1,
    seed: int = 100,
    log_freq: int = -1,
    run_inversion: bool = False,
    scheduler_kwargs: dict = {},
    negative_prompt: str = ""
):  
    num_timesteps = len(model.scheduler.timesteps)
    num_optimizer_steps = [num_recurrent_steps] * num_timesteps
    rg_start = int(rg_ratio[0] * num_timesteps)
    rg_end = int(rg_ratio[1] * num_timesteps)
    num_optimizer_steps = [o if (i >= rg_start and i < rg_end) else 0 for i, o in enumerate(num_optimizer_steps)]
    
    assert type(model.scheduler) is DDIMScheduler, "get_xt_next in diffusion_step only works with DDIM"
    print(f"Using model of type {type(model)}")
    print(f"Using rg_ratio {rg_ratio}")
    print(f"Using num_optimizer_steps {num_optimizer_steps}")

    with torch.no_grad():
        batch_size, device, dtype = latents.shape[0], latents.device, latents.dtype
        height = width = model.unet.config.sample_size * model.vae_scale_factor
        latent_height = latents.shape[-2] * model.vae_scale_factor
        latent_width = latents.shape[-1] * model.vae_scale_factor
        model_id = model.config._name_or_path
        if "xl" in model_id:
            context, added_cond_kwargs = rg_helpers.get_context_sdxl(
                model,
                prompt,
                batch_size, 
                device,
                dtype, 
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(latent_height, latent_width),
                negative_prompt=negative_prompt
            )
        else:
            context = rg_helpers.get_context(model, prompt, negative_prompt=negative_prompt)
            added_cond_kwargs = {}
        low_memory = dtype == torch.float16

    if "eta" not in scheduler_kwargs:
        scheduler_kwargs["eta"] = 1.0
    if "zs" not in scheduler_kwargs:
        if scheduler_kwargs["eta"] > 0:
            # share variance noise
            generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(batch_size)]
            scheduler_kwargs["zs"] = [rg_helpers.get_variance_noise(latents.shape, latents.device, generators).to(dtype) for _ in range(num_timesteps)]
        else:
            scheduler_kwargs["zs"] = [None for _ in range(num_timesteps)]
    seq_iter, seq_next_iter = rg_helpers.get_seq_iter(model.scheduler.timesteps, run_inversion)
    
    pbar = tqdm(enumerate(zip(seq_iter, seq_next_iter)))
    for i, (t, next_t) in pbar:
        log = log_freq > 0 and i % log_freq == 0
        latents = diffusion_step(
            model,
            controller,
            latents, 
            context,
            added_cond_kwargs,
            i, 
            t,
            next_t,
            text_weight,
            log,
            scheduler_kwargs,
            num_optimizer_steps,
            rg_weight,
            low_memory=low_memory,
            pbar=pbar
        )
    image = rg_helpers.decode_latents(model.vae, latents)
    return image, [latents, controller.gt_feat, controller.obs_feat]