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
import itertools
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
import wandb

import sys
sys.path.append("../")
from dhf.diffusion_extractor import DiffusionExtractor
from dhf.aggregation_network import AggregationNetwork

# ====================
#   Load Components
# ====================
def load_models(config_path=None, device=None, ckpt_path=None):
    if ckpt_path is None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    else:
        state_dict = torch.load(ckpt_path)
        config = state_dict["config"]
    if device is None:
        device = config.get("device", "cuda")
    diffusion_extractor = DiffusionExtractor(config, device)
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=diffusion_extractor.dims,
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"],
        **config.get("aggregation_kwargs", {})
    )
    if ckpt_path is not None:
        aggregation_network.load_state_dict(state_dict["aggregation_network"], strict=False)
    config["output_resolution"] = diffusion_extractor.output_resolution
    config["load_resolution"] = diffusion_extractor.load_resolution
    return config, diffusion_extractor, aggregation_network

def load_optimizer(config, diffusion_extractor, aggregation_network):
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]},
    ]
    if config["aggregation_kwargs"].get("use_output_head", False):
        parameter_groups.append({"params": aggregation_network.output_head.parameters(), "lr": config["lr"]})
    if config.get("low_memory", True):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(parameter_groups, weight_decay=config["weight_decay"])
    else:
        assert diffusion_extractor.dtype == torch.float32, "Regular AdamW requires torch.float32"
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    return optimizer

# ====================
#   Training Helpers
# ====================
def standardize_feats(x, y):
    # Resize y to be the same shape as x
    scale_w = x.shape[2] / y.shape[2]
    scale_h = x.shape[3] / y.shape[3]
    return torch.nn.functional.interpolate(y.clone(), scale_factor=(scale_w, scale_h))

def prepare_batch(batch, device):
    for k, v in batch.items():
        if type(batch[k]) is torch.Tensor:
            batch[k] = batch[k].to(device)
    return batch

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=False):
    with torch.inference_mode():
        feats, _ = diffusion_extractor.forward(imgs, eval_mode=eval_mode)
        b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)), diffusion_extractor.emb)
    return diffusion_hyperfeats

def save_model(config, aggregation_network, optimizer, step, run_name=None):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    if run_name is None:
        run_name = wandb.run.name
    results_folder = f"{config['results_folder']}/{run_name}"
    ckpt_folder = f"{results_folder}/ckpts"
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(dict_to_save, f"{ckpt_folder}/checkpoint_step_{step}.pt")
    OmegaConf.save(config, f"{results_folder}/config.yaml")

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    else:
        save_timestep = [0]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    return fig

def make_grid(images, nrow):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    grid = Image.fromarray(grid)
    return grid

def renormalize(x, range_a, range_b):
    # Note that if any value exceeds 255 in uint8 you get overflow
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def log_grid(imgs=None, target=None, pred=None, control_range=None):
    grid = []
    if imgs is not None:
        imgs = imgs.detach().cpu()
        imgs = renormalize(imgs, (-1, 1), (0, 1))
        grid.append(imgs)
    if target is not None:
        target = target.detach().cpu()
        target = renormalize(target, control_range, (0, 1))
        grid.append(target)
    if pred is not None:
        pred = pred.detach().cpu()
        pred = renormalize(pred, control_range, (0, 1))
        grid.append(pred)
    grid = torch.cat(grid, dim=0)
    # Clamp to prevent overflow / underflow
    grid = torch.clamp(grid, 0, 1)
    grid = make_grid(grid, imgs.shape[0])
    return grid

def make_results_folder(config, split, name, run_name=None):
    if run_name is None:
        run_name = wandb.run.name
    results_folder = f"{config['results_folder']}/{run_name}/{split}/{name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder

# ==========================
#    PointsDataset Helpers
# ==========================
def get_frame_idx(path):
    return int(os.path.basename(path).split(".")[0])

def filter_dist(data, min_dist, max_dist):
    frame_dist = lambda item: get_frame_idx(item["source"]) - get_frame_idx(item["target"])
    if min_dist > 0:
        data = [item for item in data if np.abs(frame_dist(item)) > min_dist]
    if max_dist > 0:
        data = [item for item in data if np.abs(frame_dist(item)) < max_dist]
    return data
    
def open_mask(i, video_filename, bucket_root, mask_root=None):
    if mask_root:
        mask = sorted(glob.glob(f"{bucket_root}/{mask_root}/{video_filename}/*"))
        if len(mask) > 0:
            mask = mask[i]
            mask = Image.open(mask)
        else:
            return None
    else:
        mask = None
    return mask

def open_points(i, j, video_filename, max_frame_idx, bucket_root, points_root):
    all_tracks, all_visibles = [], []
    for file in glob.glob(f"{bucket_root}/{points_root}/{video_filename}/tracks_*"):
        tracks = np.load(file)[0]
        visibles = np.load(file.replace("tracks", "visibles"))[0]
        all_tracks.append(tracks[:max_frame_idx])
        all_visibles.append(visibles[:max_frame_idx])
    if len(all_tracks) > 0:
        all_tracks = np.concatenate(all_tracks, axis=1)
        all_visibles = np.concatenate(all_visibles, axis=1)
        # Filter to visible idxs
        visible_idxs = np.logical_and(all_visibles[0], all_visibles[1])
        all_tracks = all_tracks[:, visible_idxs, :]
        all_visibles = all_visibles[:, visible_idxs]
        all_tracks, all_visibles = all_tracks[(i, j), ...], all_visibles[(i, j), ...]
    else:
        # Path does not exist
        all_tracks, all_visibles = None, None
    return all_tracks, all_visibles

def _get_valid_idxs(tracks, width, height):
    tracks_x, tracks_y = tracks[..., 0], tracks[..., 1]
    valid_x = np.logical_and(tracks_x >= 0, tracks_x < width)
    valid_y = np.logical_and(tracks_y >= 0, tracks_y < height)
    return np.logical_and(valid_x, valid_y)

def get_valid_idxs(tracks, width, height):
    # Filter for valid source, target points
    # Points may be invalid due to cropping
    valid_source = _get_valid_idxs(tracks[0], width, height)
    valid_target = _get_valid_idxs(tracks[1], width, height)
    valid_idxs = np.logical_and(valid_source, valid_target)
    return valid_idxs

def get_idxs(tracks, width):
    # Convert tracks to idxs
    tracks_x, tracks_y = tracks[..., 0], tracks[..., 1]
    # Important! We need to round at this point or errors will propagate
    tracks_x, tracks_y = tracks_x.astype(np.int32), tracks_y.astype(np.int32)
    tracks_idxs = tracks_y * width + tracks_x
    return tracks_idxs

def get_object_idxs(tracks, width, mask):
    tracks_idxs = get_idxs(tracks, width)
    source_idxs = tracks_idxs[0]
    # Determine object vs background from the source frame
    mask = np.array(mask.convert("L"))
    mask = np.where(mask == 0, 0, 1)
    mask = mask.reshape(-1)
    mask = mask[source_idxs]
    object_idxs = np.arange(0, len(source_idxs))
    object_idxs = object_idxs[mask==1]
    object_idxs = object_idxs.astype(np.int32)
    return object_idxs

def filter_tracks(tracks, width, height, mask=None, num_points=None):
    if tracks is None:
        return tracks
    valid_idxs = get_valid_idxs(tracks, width, height)
    tracks = tracks[:, valid_idxs, :]
    if mask is not None:
        object_idxs = get_object_idxs(tracks, width, mask)
        tracks = tracks[:, object_idxs, :]
    if tracks.shape[1] == 0:
        # Give one set of dummy points to avoid dataloader issues
        tracks = np.zeros((2, 1, 2))
    if num_points:
        num_tracks = tracks.shape[1]
        random_idxs = np.random.permutation(range(num_tracks))[:min(num_tracks, num_points)]
        tracks = tracks[:, random_idxs, :]
    return tracks

# ==========================
#      ControlNet Helpers
# ==========================
def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

# ==========================
#     Annotation Helpers
# ==========================
def get_nest_depth(read_root):
    return sum([x == "*" for x in read_root.split("/")])

def get_nest_prompt(path, nest_depth, prompt_file):
    k = path.split("/")[-nest_depth]
    return prompt_file[k][0]

def get_nest_name(path, nest_depth):
    return "/".join(path.split("/")[-nest_depth:])

def create_image_anns(read_root, dataset_name):
    anns = []
    for file in glob.glob(read_root):
        ann = {
            "source": os.path.basename(file),
            "dataset_name": dataset_name
        }
        anns.append(ann)
    return anns

def create_video_anns(read_root, dataset_name, bidirectional=False):
    anns = []
    for video in glob.glob(read_root):
        frame_pairs = itertools.combinations(glob.glob(f"{video}/*"), 2)
        for a, b in frame_pairs:
            video_name = os.path.basename(video)
            a = video_name + a.split(video_name)[1]
            b = video_name + b.split(video_name)[1]
            ann = {
                "source": a,
                "target": b,
                "video_name": video_name,
                "dataset_name": dataset_name
            }
            anns.append(ann)
            if bidirectional:
                ann = ann.copy()
                ann["source"] = b
                ann["target"] = a
                anns.append(ann)
    return anns

def filter_anns(anns, read_root, check_field="source"):
    nest_depth = get_nest_depth(read_root)
    pseudo_labels = set([get_nest_name(path, nest_depth) for path in glob.glob(read_root)])
    anns = [ann for ann in anns if ann[check_field] in pseudo_labels]
    return anns