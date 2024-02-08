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
# This code is taken from diffusion.py in Diffusion Hyperfeatures
# and implements diffusion sampling helpers.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/stable_diffusion/diffusion.py
# =================================================================

import torch
from diffusers import DiffusionPipeline
from dhf.stable_diffusion.resnet import set_timestep

"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""
def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    input_ids = tokens.input_ids.to(device)
    embedding = clip(input_ids).last_hidden_state
    return tokens, embedding

def get_xt_next(xt, et, at, at_next, eta):
    """
    Uses the DDIM formulation for sampling xt_next
    Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
    """
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta == 0:
        c1 = 0
    else:
        c1 = (
            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
    return x0_t, xt_next

def generalized_steps(x, model, scheduler, **kwargs):
    """
    Performs either the generation or inversion diffusion process.
    """
    seq = scheduler.timesteps
    seq = torch.flip(seq, dims=(0,))
    b = scheduler.betas
    b = b.to(x.device)

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        if kwargs.get("run_inversion", False):
            seq_iter = seq_next
            seq_next_iter = seq
        else:
            seq_iter = reversed(seq)
            seq_next_iter = reversed(seq_next)

        x0_preds = [x]
        xs = [x]
        for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
            max_i = kwargs.get("max_i", None)
            min_i = kwargs.get("min_i", None)
            if max_i is not None and i >= max_i:
                break
            if min_i is not None and i < min_i:
                continue
            
            t = (torch.ones(n) * t).to(x.device)
            next_t = (torch.ones(n) * next_t).to(x.device)
            if t.sum() == -t.shape[0]:
                at = torch.ones_like(t)
            else:
                at = (1 - b).cumprod(dim=0).index_select(0, t.long())
            if next_t.sum() == -next_t.shape[0]:
                at_next = torch.ones_like(t)
            else:
                at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
            
            # Expand to the correct dim
            at, at_next = at[:, None, None, None], at_next[:, None, None, None]
            at, at_next = at.to(x.dtype), at_next.to(x.dtype)

            if kwargs.get("run_inversion", False):
                set_timestep(model, len(seq_iter) - i - 1)
            else:
                set_timestep(model, i)

            xt = xs[-1].to(x.device)
            guidance_scale = kwargs.get("guidance_scale", -1)
            context = kwargs["context"]
            added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
            # Split into the uncond, cond half
            h = len(context)//2
            uncond, cond = context[:h], context[h:]
            uncond_kwargs = {k: v[:h] for k, v in added_cond_kwargs.items()}
            cond_kwargs = {k: v[h:] for k, v in added_cond_kwargs.items()}

            if guidance_scale == -1:
                et = model(xt, t, encoder_hidden_states=cond, added_cond_kwargs=cond_kwargs).sample
            else:
                # If using Classifier-Free Guidance, the saved feature maps
                # will be from the last call to the model, the conditional prediction
                et_uncond = model(xt, t, encoder_hidden_states=uncond, added_cond_kwargs=uncond_kwargs).sample
                et_cond = model(xt, t, encoder_hidden_states=cond, added_cond_kwargs=cond_kwargs).sample
                et = et_uncond + guidance_scale * (et_cond - et_uncond)
            
            eta = kwargs.get("eta", 0.0)
            x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta)

            x0_preds.append(x0_t)
            xs.append(xt_next.to('cpu'))

        return x0_preds

def freeze_weights(weights):
    for param in weights.parameters():
        param.requires_grad = False

def init_models(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=True,
    dtype=torch.float16
):
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    if freeze:
        freeze_weights(pipe.unet)
        freeze_weights(pipe.vae)
        freeze_weights(pipe.text_encoder)
    return pipe