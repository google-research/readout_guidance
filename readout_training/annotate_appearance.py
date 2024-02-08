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

import argparse
import glob
import json
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
from tqdm import tqdm

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline

import sys
sys.path.append("../")
from readout_guidance import rg_helpers
from readout_training import train_helpers

def load_img2img_pipeline(config, device, dtype=torch.float16):
    if "stable-diffusion-xl" in config["model_path"]:
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(config["model_path"], torch_dtype=dtype).to(device)
    else:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(config["model_path"], torch_dtype=dtype).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    rg_helpers.load_scheduler(pipeline, config["model_path"], mode="ddim")
    return pipeline, dtype

def main(args, device="cuda"):
    config = OmegaConf.load(args.config_path)
    batch_size = config["batch_size"]
    prompt_file = json.load(open(config["prompt_file"]))
    
    pipeline, _ = load_img2img_pipeline(config, device)
    height = width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    image_dim = (width, height)

    image_paths = glob.glob(config["read_root"])
    nest_depth = train_helpers.get_nest_depth(config["read_root"])
    for i in tqdm(range(0, len(image_paths), batch_size)):
        # Chunk anns into groups of size batch_size
        chosen_paths = image_paths[i:i+batch_size]
        prompts = [train_helpers.get_nest_prompt(path, nest_depth, prompt_file) for path in chosen_paths]
        images = [Image.open(path) for path in chosen_paths]
        original_size = [image.size for image in images]
        images = [image.resize(image_dim) for image in images]
        with torch.inference_mode():
            outputs = pipeline(prompt=prompts, image=images, target_size=image_dim, **config["pipe_kwargs"])[0]
            outputs = [outputs[i].resize(original_size[i]) for i in range(len(outputs))]
        save_folder = config["save_root"]
        save_folder = f"{save_folder}/strength-{config['pipe_kwargs']['strength']}_gs-{config['pipe_kwargs']['guidance_scale']}"
        for output, path in zip(outputs, chosen_paths):
            save_name = train_helpers.get_nest_name(path, nest_depth)
            save_name = f"{save_folder}/{save_name}"
            save_folder_ann = os.path.dirname(save_name)
            if not os.path.exists(save_folder_ann):
                os.makedirs(save_folder_ann)
            output.save(save_name)

if __name__ == '__main__':
    # conda run -n readout python3 annotate_appearance.py --config_path configs/annotate_appearance.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)