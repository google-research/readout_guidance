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

import glob
import json
from io import BytesIO
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import mediapy
import numpy as np
from PIL import Image
import sys
import torch
from tqdm import tqdm

from readout_guidance import rg_helpers, rg_operators

# ======================
#     Edits Helpers
# ======================
def set_edits_points(edits, points):
    for edit in edits:
       if edit["head_type"] == "correspondence":
            edit["points"] = points
    return edits

def set_edits_image(edits, image):
    for edit in edits:
        if edit["head_type"] == "appearance":
            edit["control_image"] = image
    return edits

# ======================
#     Points Helpers
# ======================
def interpolate_points(points, alpha):
    points1, points2 = points[0].copy(), points[1].copy()
    dx = points2[:, 0] - points1[:, 0]
    dy = points2[:, 1] - points1[:, 1]
    points2 = points1.copy()
    points2[:, 0] = points2[:, 0] + alpha * dx
    points2[:, 1] = points2[:, 1] + alpha * dy
    points = np.stack([points1, points2])
    return points

def latent_points(tracks, visibles, image_dim, latent_dim):
    tracks, visibles = tracks[0], visibles[0]
    f, n, c = tracks.shape
    tracks = tracks[..., (1, 0)]
    tracks = rg_operators.rescale_points(tracks, image_dim, latent_dim)
    tracks = torch.from_numpy(tracks)
    return tracks, visibles

def open_points(tracks_file, ann, latent_dim, tracks_idx=0, interp_num=None):
    tracks_name = tracks_file[tracks_idx]
    visibles_file = tracks_name.replace("tracks", "visibles")
    tracks = np.load(tracks_name)
    if os.path.exists(visibles_file):
        visibles = np.load(visibles_file)
    else:
        visibles = np.ones((1, 2, tracks.shape[-2]))
    # Interpolate for image animation
    if tracks.shape[1] == 2 and interp_num:
        interp_tracks = []
        interp_visibles = []
        for alpha in np.linspace(0, 1, interp_num):
            interp_tracks.append(interpolate_points(tracks[0], alpha)[1][None, None, ...])
            interp_visibles.append(visibles[:, 0, :][:, None, :])
        tracks = np.concatenate(interp_tracks, axis=1)
        visibles = np.concatenate(interp_visibles, axis=1)
    # Convert points from (x, y) to (y, x)
    image_dim = ann["image_dim"][::-1]
    original_tracks, original_visibles = tracks, visibles
    tracks, visibles = latent_points(tracks, visibles, image_dim, latent_dim)
    return tracks_name, tracks, visibles, original_tracks, original_visibles

def subselect_points(points, n):
    points_idxs = np.random.permutation(range(points.shape[1]))
    points_idxs = points_idxs[:min(n, len(points_idxs))]
    points = points[:, points_idxs, :]
    return points

def viz_tracks(first_frame, tracks, latent_dim, save_file=None):
    tracks = tracks.detach().clone()
    tracks = rg_operators.rescale_points(tracks, latent_dim, first_frame.size[::-1])
    tracks = tracks[..., (1, 0)] # (y, x) -> (x, y)
    tracks = tracks[(0, -1), :, :] # select first and last frame
    fig, ax = plt.subplots()
    # Fade the image
    ax.imshow(first_frame, alpha=0.3)
    for i in range(tracks.shape[1]):
        arrow = FancyArrowPatch((tracks[0, i, 0], tracks[0, i, 1]), (tracks[1, i, 0], tracks[1, i, 1]),
                            arrowstyle='->', color='red', mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)
    ax.axis("off")
    if save_file is None:
        buffer = BytesIO()
        save_file = buffer
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    img = Image.open(save_file).convert("RGB")
    return img

# ======================
#   Generation Helpers
# ====================== 
def open_first_frame(tracks_file):
    first_frame_file = f"{os.path.dirname(tracks_file[0])}/source.png"
    if os.path.exists(first_frame_file):
        first_frame = Image.open(first_frame_file)
    else:
        first_frame = None
    return first_frame

def create_frames(
    pipeline, 
    prompts,
    latents,
    edits, 
    latent_dim, 
    config, 
    tracks, 
    visibles,
    num_frames,
    first_frame
):
    num_frames = min(num_frames, tracks.shape[0])
    predicted_frames = []
    for i in range(1, num_frames):
        frame_latents = latents.detach().clone()
        # Select points covisible pairs of frames
        points = tracks[(0, i), :, :]
        visible_points = visibles[(0, i), :].all(axis=0)
        points = points[:, visible_points, :]
        edits = set_edits_points(edits, points)
        edits = set_edits_image(edits, first_frame)
        images, _ = rg_helpers.run_preset_generation(
            pipeline, 
            prompts, 
            frame_latents, 
            edits, 
            latent_dim=latent_dim,
            **config["generation_kwargs"]
        )
        if len(predicted_frames) == 0:
            predicted_frames.append(images[0])
        predicted_frames.append(images[1])
    return predicted_frames

def run_ddim_inversion(config, pipeline, first_frame, prompt, image_dim, dtype, batch_size):
    with torch.no_grad():
        generation_kwargs = {k: v for k, v in config["generation_kwargs"].items() if k != "text_weight"}
        guidance_scale = config["generation_kwargs"]["text_weight"]
        if type(guidance_scale) is not float:
            guidance_scale = guidance_scale[0]
        _, inverted_latents = rg_helpers.run_preset_inversion(
            pipeline, 
            first_frame, 
            prompt, 
            image_dim=image_dim,
            dtype=dtype,
            text_weight=guidance_scale,
            **generation_kwargs
        )
        latents = torch.cat([inverted_latents[0].detach().clone() for _ in range(batch_size)], dim=0)
        latents = latents.to(dtype)
    return latents

def main(config_path, device="cuda"):
    config = OmegaConf.load(config_path)
    assert config.get("same_seed", True) is True, "Only same_seed currently supported"
    
    # Load pipeline
    pipeline, dtype = rg_helpers.load_pipeline(config, device)
    batch_size = config["batch_size"]
    latent_height = latent_width = pipeline.unet.config.sample_size
    height = width = latent_height * pipeline.vae_scale_factor
    image_dim = (width, height)
    latent_dim = (latent_height, latent_width)

    # Create root save folder
    save_folder = config["output_dir"]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    OmegaConf.save(config, f"{save_folder}/config.yaml")

    # Create edit config and load aggregation network
    num_frames = config.get("num_frames", 2)
    edits = rg_helpers.get_edits(config, device, dtype)
    ann_paths = glob.glob(f"{config['points_root']}/*/info.json")

    for ann_path in tqdm(ann_paths):
        ann = json.load(open(ann_path))
        name, prompt = ann["video_name"], ann["prompt"]
        tracks_file = list(sorted(glob.glob(f"{config['points_root']}/{ann['video_name']}/tracks*")))
        first_frame = open_first_frame(tracks_file)
        first_frame = first_frame.resize(image_dim)
        seed = ann["seed"]
        prompts, latents = rg_helpers.get_prompts_latents(
            pipeline,
            prompt,
            batch_size, 
            seed,
            latent_dim,
            device,
            dtype,
        )
        if config.get("run_ddim_inversion", False):
            latents = run_ddim_inversion(
                config, 
                pipeline, 
                first_frame, 
                prompt, 
                image_dim, 
                dtype, 
                batch_size
            )
        for tracks_idx in range(len(tracks_file)):
            tracks_name, tracks, visibles, original_tracks, original_visibles = open_points(
                tracks_file, 
                ann, 
                latent_dim, 
                tracks_idx=tracks_idx, 
                interp_num=num_frames
            )
            predicted_frames = create_frames(pipeline, 
                prompts, 
                latents, 
                edits,
                latent_dim, 
                config, 
                tracks, 
                visibles,
                num_frames,
                first_frame
            )
            # Create tracks folder
            save_name = os.path.basename(tracks_name).split("_")[1].split(".")[0]
            ann_name = ann['video_name'].split('.')[0]
            if name != ann_name:
                save_name = name + "_" + save_name
            drag_folder = f"{save_folder}/{ann_name}/{save_name}"
            if not os.path.exists(drag_folder):
                os.makedirs(drag_folder, exist_ok=True)
            # Save tracks info
            viz_tracks(first_frame, tracks, latent_dim, f"{drag_folder}/tracks.png")
            np.save(f"{drag_folder}/tracks.npy", original_tracks)
            np.save(f"{drag_folder}/visibles.npy", original_visibles)
            # Save frames
            for i, frame in enumerate(predicted_frames):
                Image.fromarray(frame).save(f"{drag_folder}/rg_{str(i).zfill(5)}.png")
            # Prepend the first frame and tracks
            tracks_frame = np.array(viz_tracks(first_frame, tracks, latent_dim).resize(image_dim))
            predicted_frames = [predicted_frames[0], tracks_frame] + predicted_frames
            mediapy.write_video(f"{drag_folder}/rg.gif", np.stack(predicted_frames), fps=config["fps"], codec="gif")
        
if __name__ == "__main__":
    # python3 script_drag.py configs/drag_real.yaml
    config_path = sys.argv[1]
    main(config_path)