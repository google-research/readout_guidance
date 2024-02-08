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
import einops
from io import BytesIO
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import wandb

import sys
sys.path.append("../")
from dhf.correspondence_utils import (
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    compute_pck,
    rescale_points
)
from script_drag import viz_tracks
from readout_training.dataset import PointsDataset
from readout_training import train_helpers

def get_rescale_size(config):
    output_size = (config["output_resolution"], config["output_resolution"])
    if "load_resolution" in config:
        load_size = (config["load_resolution"], config["load_resolution"])
    else:
        load_size = output_size
    return output_size, load_size

def load_points_batch(ann, output_size, device):
    def to_pil(x):
        x = einops.rearrange(x, 'c w h -> w h c')
        x = x.detach().cpu().numpy()
        x = (x + 1) / 2
        x = x * 255.0
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        return x
    ann = train_helpers.prepare_batch(ann, device)
    points = ann["points"][0]
    # Permute from (x, y) to (y, x)
    points = points[..., (1, 0)]
    points[..., 0] = points[..., 0] * output_size[0]
    points[..., 1] = points[..., 1] * output_size[1]
    source_points = points[0].detach().cpu().numpy()
    target_points = points[1].detach().cpu().numpy()
    imgs = torch.cat([ann["source"], ann["target"]], dim=0)
    img1_pil = to_pil(imgs[0])
    img2_pil = to_pil(imgs[1])
    return source_points, target_points, img1_pil, img2_pil, imgs

# ====================
#     Dataloader
# ====================
def get_correspondence_loader(config, annotation_file, shuffle):
    dataset = PointsDataset(
        annotation_file, 
        size=config["load_resolution"],
        **config["dataset_args"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"]
    )
    return dataset, dataloader

# ====================
#        Loss
# ====================
def loss_correspondence(aggregation_network, hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are this shape to avoid complex indexing
    img1_hyperfeats, img2_hyperfeats = hyperfeats[0][None, ...], hyperfeats[1][None, ...]
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss

def viz_tracks_pred_target(
    img1_pil,
    img2_pil,
    source_points, 
    target_points,
    predicted_points, 
    source_size, 
    target_size, 
    load_size, 
    title,
    save_file=None,
    num_points_max=20
):  
    num_points = source_points.shape[0]
    idxs = np.random.permutation(range(num_points))[:min(num_points_max, num_points)]
    source_points =  rescale_points(source_points, source_size, load_size)
    predicted_points = rescale_points(predicted_points, target_size, load_size)
    target_points = rescale_points(target_points, target_size, load_size)
    predicted_tracks = torch.from_numpy(np.stack([source_points[idxs], predicted_points[idxs]]))
    target_tracks = torch.from_numpy(np.stack([source_points[idxs], target_points[idxs]]))
    predicted_img = viz_tracks(img1_pil, predicted_tracks, load_size)
    plt.clf()
    target_img = viz_tracks(img1_pil, target_tracks, load_size)
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].set_title("frame 1")
    axes[0, 0].imshow(img1_pil)
    axes[0, 1].set_title("frame 2")
    axes[0, 1].imshow(img2_pil)
    axes[1, 0].set_title("target")
    axes[1, 0].imshow(target_img)
    axes[1, 1].set_title("predicted")
    axes[1, 1].imshow(predicted_img)
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    plt.suptitle(title)
    if save_file is None:
        buffer = BytesIO()
        save_file = buffer
    fig.savefig(save_file, bbox_inches='tight', pad_inches=0)
    img = Image.open(save_file).convert("RGB")
    return img

# ====================
#  Validate and Train
# ====================
def validate(config, diffusion_extractor, aggregation_network, val_dataloader, global_step):
    device = config.get("device", "cuda")
    log_max = config.get("log_max", None)
    plot_every_n_steps = config.get("plot_every_n_steps", -1)
    pck_threshold = config["pck_threshold"]
    output_size, load_size = get_rescale_size(config)

    val_pck_img = []
    for j, ann in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            if log_max is None or j < log_max:
                source_points, target_points, img1_pil, img2_pil, imgs = load_points_batch(ann, load_size, device)
                source_size, target_size = load_size, load_size
                hyperfeats = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
                # Compute loss
                loss = loss_correspondence(aggregation_network, hyperfeats, source_points, target_points, output_size)
                # Compute nearest neighbors
                img1_hyperfeats, img2_hyperfeats = hyperfeats[0][None, ...], hyperfeats[1][None, ...]
                _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
                predicted_points = predicted_points.detach().cpu().numpy()
                # Compute PCK rescaled to the original image dimensions
                predicted_points = rescale_points(predicted_points, load_size, target_size)
                target_points = rescale_points(target_points, load_size, target_size)
                dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
                # Save metrics and loss
                val_pck_img.append(pck_img)
                wandb.log({"val/loss": loss.item()}, step=j)
                if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
                    title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
                    results_folder = train_helpers.make_results_folder(config, "val", "preds")
                    save_file = f"{results_folder}/step-{global_step}_b-{j}.png"
                    viz_tracks_pred_target(
                        img1_pil,
                        img2_pil,
                        source_points, 
                        target_points,
                        predicted_points, 
                        source_size, 
                        target_size,
                        load_size, 
                        title,
                        save_file
                    )
            else:
                break
    val_pck_img = np.concatenate(val_pck_img)
    wandb.log({"val/pck_img": val_pck_img.sum() / len(val_pck_img)})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader):
    device = config.get("device", "cuda")
    run_name = config["id"]
    max_steps = config["max_steps"]
    max_epochs = math.ceil(max_steps / len(train_dataloader))
    
    output_size, load_size = get_rescale_size(config)
    np.random.seed(0)

    aggregation_network = aggregation_network.to(device)
    train_end = False
    global_step = 0
    for epoch in range(max_epochs):
        for i, ann in tqdm(enumerate(train_dataloader)):
            source_points, target_points, _, _, imgs = load_points_batch(ann, output_size, device)
            hyperfeats = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = loss_correspondence(aggregation_network, hyperfeats, source_points, target_points, output_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()}, step=global_step)
            wandb.log({"train/diffusion_timestep": diffusion_extractor.save_timestep[0]}, step=global_step)
            if global_step > 0 and config["val_every_n_steps"] > 0 and global_step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    fig = train_helpers.log_aggregation_network(aggregation_network, config)
                    wandb.log({f"mixing_weights": fig})
                    train_helpers.save_model(config, aggregation_network, optimizer, global_step, run_name=run_name)
                    validate(config, diffusion_extractor, aggregation_network, val_dataloader, global_step)
            global_step += 1
            if max_steps > 0 and global_step > max_steps:
                train_end = True
                break
        if train_end:
            break

def main(args):
    config, diffusion_extractor, aggregation_network = train_helpers.load_models(args.config_path)
    
    wandb.init(**config["wandb_kwargs"])
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    run_name = wandb.run.name
    
    optimizer = train_helpers.load_optimizer(config, diffusion_extractor, aggregation_network)
    train_dataset, train_dataloader = get_correspondence_loader(config, config["train_file"], True)
    val_dataset, val_dataloader = get_correspondence_loader(config, config["val_file"], False)
    
    config["id"] = run_name
    config["dims"] = diffusion_extractor.dims
    assert config["batch_size"] == 1, "loss_correspondence only supports batch_size=1"
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # python3 train_correspondence.py --config_path configs/train_correspondence.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)