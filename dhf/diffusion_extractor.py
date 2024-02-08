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

# =================================================================
# This code is adapted from diffusion_extractor.py in Diffusion Hyperfeatures
# and implements extraction of diffusion features from a single random timestep 
# of the generation process, rather than all features from the inversion process.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/diffusion_extractor.py
# =================================================================

import ast
import einops
import numpy as np
import torch

from dhf.stable_diffusion.diffusion import generalized_steps
from dhf.stable_diffusion.resnet import init_resnet_func, collect_feats, collect_channels
from readout_guidance import rg_helpers

class DiffusionExtractor:
    def __init__(self, config, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.model_id = config["model_id"]
        self.model, self.dtype = rg_helpers.load_pipeline({"model_path": self.model_id}, device, dtype=self.dtype)
        self.unet, self.vae = self.model.unet, self.model.vae
        # Timestep scheduling
        self.scheduler = self.model.scheduler
        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.emb = None
        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [0])
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))
        self.batch_size = config.get("batch_size", 1)
        # Text Embeddings
        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        # Automatically determine default latent and image dim
        self.height = self.width = self.unet.config.sample_size * self.model.vae_scale_factor
        self.latent_height = self.latent_width = self.unet.config.sample_size
        self.load_resolution = self.height
        self.output_resolution = self.latent_height
        # Hyperparameters
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        self.save_mode = config.get("save_mode", "hidden")
        self.eval_mode = config.get("eval_mode", False)
        if "idxs" in config and config["idxs"] is not None:
            self.idxs = ast.literal_eval(config["idxs"])
        else:
            self.idxs = None
        self.dims = collect_channels(self.unet, idxs=self.idxs)
        print(f"diffusion_mode: {self.diffusion_mode}")
        print(f"idxs: {self.idxs}")
        print(f"output_resolution: {self.output_resolution}")
        print(f"prompt: {self.prompt}")
        print(f"negative_prompt: {self.negative_prompt}")
        print(f"save_mode: {self.save_mode}")

    def change_cond(self, prompt, negative_prompt, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        with torch.no_grad():
            if "xl" in self.model_id:
                context, added_cond_kwargs = rg_helpers.get_context_sdxl(
                    self.model,
                    [prompt] * batch_size,
                    batch_size, 
                    self.device,
                    self.dtype,
                    original_size=(self.height, self.width),
                    crops_coords_top_left=(0, 0),
                    target_size=(self.height, self.width),
                    negative_prompt=negative_prompt
                )
            else:
                context = rg_helpers.get_context(self.model, [prompt] * batch_size, negative_prompt=negative_prompt)
                added_cond_kwargs = {}
        return context, added_cond_kwargs

    def random_latents(self):
        return torch.randn(
            (   
                self.batch_size, 
                self.unet.in_channels, 
                self.latent_height, 
                self.latent_width
            ), 
            dtype=self.dtype, 
            device=self.device, 
            generator=self.generator
        )

    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        context, added_cond_kwargs = self.change_cond(self.prompt, self.negative_prompt, latent.shape[0])
        xs = generalized_steps(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False, 
            guidance_scale=guidance_scale,
            context=context,
            min_i=min_i,
            max_i=max_i,
            added_cond_kwargs=added_cond_kwargs
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        if not preview_mode:
            init_resnet_func(self.unet, save_mode=self.save_mode, reset=True, idxs=self.idxs)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = rg_helpers.collect_and_resize_feats(self.model, self.idxs, self.output_resolution)
            # convert feats to [batch_size, num_timesteps, channels, w, h]
            feats = feats[..., None]
            feats = einops.rearrange(feats, 'b w h l s -> b s l w h')
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    def forward(self, images, guidance_scale=-1, preview_mode=False, eval_mode=False):
        assert self.diffusion_mode == "generation", "Only generation, not inversion, supported."
        latents = rg_helpers.images_to_latents(self.vae, images, (self.height, self.width), self.dtype)
        if eval_mode or self.eval_mode:
            self.save_timestep = [self.num_timesteps-1]
        else:
            self.save_timestep = [np.random.choice(range(1, self.num_timesteps))]
            noise_timestep = self.num_timesteps - 1 - self.save_timestep[0]
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, torch.tensor(noise_timestep))
        save_timestep = self.save_timestep[0]
        self.emb = rg_helpers.embed_timestep(self.unet, latents, save_timestep)
        extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, min_i=save_timestep, max_i=save_timestep+1)

        with torch.no_grad():
            return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)