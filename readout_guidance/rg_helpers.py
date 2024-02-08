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

import einops
import json
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from diffusers import (
    DDIMScheduler,
    DDPMScheduler, 
    PNDMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    T2IAdapter, 
    StableDiffusionXLAdapterPipeline
)

from dhf.aggregation_network import AggregationNetwork
from dhf.stable_diffusion.resnet import collect_feats
from readout_guidance import rg_operators, rg_pipeline, rg_helpers

# ====================
#   Load Components
# ====================
def load_pipeline(config, device, dtype=None, scheduler_mode="ddim"):
    if "xl" in config["model_path"]:
        dtype = torch.float16 if dtype is None else dtype
        pipeline = StableDiffusionXLPipeline.from_pretrained(config["model_path"], torch_dtype=dtype).to(device)
    else:
        dtype = torch.float32 if dtype is None else dtype
        pipeline = StableDiffusionPipeline.from_pretrained(config["model_path"], torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    load_scheduler(pipeline, config["model_path"], mode=scheduler_mode)
    return pipeline, dtype

def load_controlnet_pipeline(config, device, dtype=torch.float32, scheduler_mode="ddim"):
    controlnet = ControlNetModel.from_pretrained(config["controlnet_path"], torch_dtype=dtype)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(config["model_path"], controlnet=controlnet, torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    load_scheduler(pipeline, config["model_path"], mode=scheduler_mode)
    return pipeline, dtype

def load_adapter_pipeline(config, device, dtype=torch.float16, scheduler_mode="ddim"):
    t2i_adapter = T2IAdapter.from_pretrained(config["adapter_path"], torch_dtype=dtype)
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(config["model_path"], adapter=t2i_adapter, torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    load_scheduler(pipeline, config["model_path"],  mode=scheduler_mode)
    return pipeline, dtype

def load_scheduler(pipeline, MODEL_ID, mode="ddim"):
    if mode == "ddim":
        scheduler_cls = DDIMScheduler
    elif mode == "ddpm":
        scheduler_cls = DDPMScheduler
    elif mode == "pndm":
        scheduler_cls = PNDMScheduler
    elif mode == "ead":
        scheduler_cls = EulerAncestralDiscreteScheduler
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

def load_aggregation_network(aggregation_config, device, dtype):
    weights_path = aggregation_config["aggregation_ckpt"]
    state_dict = torch.load(weights_path)
    config = state_dict["config"]
    aggregation_kwargs = config.get("aggregation_kwargs", {})
    custom_aggregation_kwargs = {k: v for k, v in aggregation_config.items() if "aggregation" not in k}
    aggregation_kwargs = {**aggregation_kwargs, **custom_aggregation_kwargs}
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=config["dims"],
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"],
        **aggregation_kwargs
    )
    aggregation_network.load_state_dict(state_dict["aggregation_network"], strict=False)
    aggregation_network = aggregation_network.to(device).to(dtype)
    return aggregation_network, config

# ====================
#   Load Latents
# ====================
def get_latents(pipeline, batch_size, device, generator, dtype, latent_dim):
    latents_shape = (pipeline.unet.in_channels, *latent_dim)
    latents = torch.randn((batch_size, *latents_shape), generator=generator)
    latents = latents.to(device).to(dtype)
    return latents

def get_prompts_latents_video(prompts, latents):
    return [prompts[0]], einops.rearrange(latents, 'f c h w -> 1 c f h w')

def get_prompts_latents(pipeline, prompt, batch_size, seed, latent_dim, device, dtype=torch.float32, same_seed=True):
    generator = torch.Generator().manual_seed(seed)
    latents_shape = (pipeline.unet.in_channels, *latent_dim)
    prompts = [prompt] * batch_size
    if same_seed:
        latents = torch.randn((1, *latents_shape), generator=generator)
        latents = latents.repeat((batch_size, 1, 1, 1))
    else:
        latents = torch.randn((batch_size, *latents_shape), generator=generator)
    latents = latents.to(device).to(dtype)
    return prompts, latents

def load_correspondences(file, latent_dim, frame_idx):
    # Note that points should be in (y, x) format since everything is (h, w) order
    correspondences = json.load(open(file))
    image_dim, points = correspondences["image_dim"], torch.Tensor(correspondences[str(frame_idx)])
    points = rg_operators.rescale_points(points, image_dim, latent_dim)
    return points

# =========================================
#    Features, Latents, and Text Context
# =========================================
def resize(x, old_res, new_res, mode):
    # (batch_size, width * height, channels)
    batch_size, size, channels = x.shape
    x = x.reshape((batch_size, *old_res, channels))
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = torch.nn.functional.interpolate(x, size=new_res, mode=mode)
    x = einops.rearrange(x, 'b c h w -> b h w c')
    return x

def resize_feat(feat, new_res, resize_mode="bilinear"):
    old_res = feat.shape[2:]
    feat = einops.rearrange(feat, 'b c h w -> b (h w) c')
    feat = resize(feat, old_res=old_res, new_res=new_res, mode=resize_mode)
    return feat

def collect_and_resize_feats(model, idxs, latent_dim):
    if model is None:
        return None
    feature_store = {"up": collect_feats(model.unet, idxs=idxs)}
    feats = []
    for key in feature_store:
        for i, feat in enumerate(feature_store[key]):
            feat = rg_helpers.resize_feat(feat, new_res=latent_dim)
            feats.append(feat)
    # Concatenate all layers along the channel
    # dimension to get shape (b s d)
    if len(feats) > 0:
        feats = torch.cat(feats, dim=-1)
    else:
        feats = None
    return feats

def image_to_tensor(image):
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = image[None, ...]
    image = einops.rearrange(image, 'b w h c -> b c w h')
    image = torch.from_numpy(image)
    image = image / 255.0
    image = 2. * image - 1.
    return image

def images_to_latents(vae, image, image_dim, dtype):
    with torch.no_grad():
        # Run vae in torch.float32 always to avoid black images
        vae = vae.to(torch.float32)
        if not torch.is_tensor(image):
            image = image_to_tensor(image)
        image = image.to(vae.device).to(vae.dtype)
        image = torch.nn.functional.interpolate(image, size=(image_dim[0], image_dim[1]), mode="bilinear")
        latents = vae.encode(image).latent_dist.sample(generator=None) 
        latents = latents * vae.config.scaling_factor
        latents = latents.to(dtype)
    return latents

def decode_latents(vae, latents):
    # Ensure that vae is always in torch.float32 to prevent black images / underflow
    vae = vae.to(torch.float32)
    latents = latents.to(torch.float32)
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def get_context(model, prompt, negative_prompt=""):
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [negative_prompt] * len(prompt), padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    return context

def get_context_sdxl(
    model, 
    prompt,
    batch_size, 
    device,
    dtype,
    original_size=(1024, 1024),
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
    negative_prompt=""
):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = model.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        negative_prompt=[negative_prompt] * len(prompt)
    )
    context = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    add_time_ids = model._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    add_time_ids = add_time_ids.repeat((batch_size, 1))
    context = context.to(device).to(dtype)
    add_text_embeds = add_text_embeds.to(device).to(dtype)
    add_time_ids = add_time_ids.to(device).to(dtype)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return context, added_cond_kwargs

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    images = [np.array(image) for image in images]
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

# ========================
#    Scheduler Updates
# ========================
def get_variance_noise(shape, device, generator=None):
    if generator:
        variance_noise = [torch.randn((1, *shape[1:]), device=device, generator=g) for g in generator]
        return torch.vstack(variance_noise)
    else:
        return torch.randn(shape, device=device)
    
def get_seq_iter(timesteps, run_inversion):
    seq = timesteps
    seq = torch.flip(seq, dims=(0,))
    seq_next = [-1] + list(seq[:-1])
    if run_inversion:
        seq_iter = seq_next
        seq_next_iter = seq
    else:
        seq_iter = reversed(seq)
        seq_next_iter = reversed(seq_next)
    return seq_iter, seq_next_iter

def get_at_next(scheduler, t, next_t, et):
    get_at = lambda t: scheduler.alphas_cumprod[t] if t != -1 else scheduler.final_alpha_cumprod
    get_at_next = lambda next_t: scheduler.alphas_cumprod[next_t] if next_t != -1 else scheduler.final_alpha_cumprod
    if type(t) is int or len(t.shape) == 0:
        at = get_at(t)
        at_next = get_at_next(next_t)
    else:
        device, dtype = et.device, et.dtype
        at = torch.tensor([get_at(_t) for _t in t[:et.shape[0]]])
        at_next = torch.tensor([get_at_next(_next_t) for _next_t in next_t[:et.shape[0]]])
        at = at[:, None, None, None].to(device).to(dtype)
        at_next = at_next[:, None, None, None].to(device).to(dtype)
    return at, at_next

def get_xt_next(scheduler, et, t, next_t, xt, eta=0.0, variance_noise=None):
    """
    Uses the DDIM formulation for sampling xt_next
    Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
    """
    at, at_next = get_at_next(scheduler, t, next_t, et)
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta > 0:
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    else:
        c1 = 0
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x0_t + c2 * et
    if eta > 0:
        if variance_noise is not None:
            xt_next = xt_next + c1 * variance_noise
        else:
            xt_next = xt_next + c1 * torch.randn_like(xt_next)
    return xt_next, x0_t

# ========================
#  Timestep Conditioning
# ========================
def preprocess_timestep(sample, timestep):
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    return timesteps

def embed_timestep(unet, sample, timestep):
    timesteps = preprocess_timestep(sample, timestep)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb, None)
    return emb

# ========================
#    Preset RG Pipeline
# ========================
def get_edits(config, device, dtype):
    edits = []
    for item in config["rg_kwargs"]:
        item = OmegaConf.to_container(item, resolve=True)
        aggregation_network, aggregation_config = load_aggregation_network(item["aggregation_kwargs"], device, dtype)
        item["aggregation_network"] = aggregation_network
        item["aggregation_kwargs"] = {**item["aggregation_kwargs"], **aggregation_config}
        edits.append(item)
    return edits

def run_preset_generation(
    pipeline, 
    prompts, 
    latents, 
    edits=[],
    # generation kwargs
    text_weight=7.5,
    rg_weight=2e-2,
    rg_ratio=[0.0, 1.0],
    num_recurrent_steps=1,
    negative_prompt="",
    log_freq=10,
    # scheduler kwargs
    eta=1.0,
    num_timesteps=100,
    zs=None,
    # rg kwargs
    idxs=None,
    points=None,
    latent_dim=None
):
    scheduler_kwargs = {"eta": eta}
    if zs is not None:
        scheduler_kwargs["zs"] = zs
    controller = rg_pipeline.ReadoutGuidance(
        pipeline,
        edits=edits,
        points=points,
        latent_dim=latent_dim,
        idxs=idxs
    )
    pipeline.scheduler.set_timesteps(num_timesteps)
    return rg_pipeline.text2image_rg(
        pipeline, 
        controller, 
        prompts, 
        latents,
        text_weight=text_weight,
        rg_weight=rg_weight, 
        rg_ratio=rg_ratio,
        num_recurrent_steps=num_recurrent_steps,
        log_freq=log_freq,
        scheduler_kwargs=scheduler_kwargs,
        negative_prompt=negative_prompt
    )

def run_preset_inversion(
    pipeline, 
    image, 
    prompt="",
    log_freq=10,
    text_weight=0.0, 
    num_timesteps=50,
    image_dim=None,
    dtype=None,
    **kwargs
):
    latents = rg_helpers.images_to_latents(pipeline.vae, image, image_dim, dtype)
    scheduler_kwargs = {"eta": 0.0}
    controller = rg_pipeline.ReadoutGuidance()
    pipeline.scheduler.set_timesteps(num_timesteps)
    return rg_pipeline.text2image_rg(
        pipeline, 
        controller,
        [prompt], 
        latents, 
        text_weight=text_weight, 
        log_freq=log_freq, 
        scheduler_kwargs=scheduler_kwargs, 
        run_inversion=True
    )