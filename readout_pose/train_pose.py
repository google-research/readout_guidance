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
import math
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import wandb

import sys
sys.path.append("../")
from readout_training import train_helpers
from readout_pose import pose_helpers
from readout_pose.dataset import MSCOCODataset

# ====================
#     Dataloader
# ====================
def get_mscoco_loader(config, annotation_file, shuffle):
    dataset = MSCOCODataset(
        annotation_file,
        size=config["load_resolution"],
        **config["dataset_args"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    return dataset, dataloader

# ====================
#        Loss
# ====================
def loss_pose(config, aggregation_network, pred, target):
    loss = torch.nn.functional.mse_loss(pred, target)
    return loss

# ====================
#    Visualizations
# ====================
def draw_pose(res, meta):
    candidate, subset = meta["candidate"], meta["subset"]
    candidate, subset = np.array(candidate), np.array(subset)
    canvas = np.zeros((res[1], res[0], 3)).astype(np.uint8)
    canvas = pose_helpers.draw_bodypose(canvas, candidate, subset)
    canvas = Image.fromarray(canvas).convert("RGB")
    return canvas

def heatmap_to_meta(imgs, heatmap, thresh=0.1):
    size, latent_size = imgs.shape[-1], heatmap.shape[-1]
    candidate, subset = pose_helpers.heatmap_to_meta(heatmap, size, latent_size, thresh=thresh)
    return [{"candidate": candidate[b], "subset": subset[b]} for b in range(len(candidate))]

def draw_meta(imgs, meta):
    size = imgs.shape[-1]
    canvas = [draw_pose((size, size), m) for m in meta]
    canvas = [torch.from_numpy(np.array(c)) for c in canvas]
    canvas = [einops.rearrange(c, 'h w c -> c h w') for c in canvas]
    return torch.stack(canvas)

def update_pcks(pcks, pred_meta, target_meta):
    for thr in pcks:
        for i in range(len(pred_meta)):
            pred_keypoints = pose_helpers.openpose_to_mscoco(pred_meta[i])
            gt_keypoints = pose_helpers.openpose_to_mscoco(target_meta[i])
            acc = pose_helpers.compute_pose_pck(pred_keypoints, gt_keypoints, thr)
            if acc is not None:
                pcks[thr].append(acc)
    return pcks

# ====================
#  Validate and Train
# ====================
def validate(config, diffusion_extractor, aggregation_network, dataloader, split, step, run_name):
    device = aggregation_network.device
    total_loss = []
    pcks = {thresh: [] for thresh in [0.05, 0.1, 0.2]}
    for j, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = train_helpers.prepare_batch(batch, device)
            imgs, target = batch["source"], batch["heatmap"]
            # Compute Loss
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
            loss = loss_pose(config, aggregation_network, pred, target)
            total_loss.append(loss.item())
            # Compute PCK
            target_meta = heatmap_to_meta(imgs, target)
            pred_meta = heatmap_to_meta(imgs, pred)
            pcks = update_pcks(pcks, pred_meta, target_meta)
            log_max = config.get("log_max")
            if log_max == -1 or j < log_max:
                grid = train_helpers.log_grid(imgs, draw_meta(imgs, target_meta), draw_meta(imgs, pred_meta), (0, 255))
                results_folder = train_helpers.make_results_folder(config, split, "preds", run_name=run_name)
                grid.save(f"{results_folder}/step-{step}_b-{j}.png")
            else:
                if split == "train":
                    break
    if split == "val":
        wandb.log({f"{split}/loss": loss})
        for thr, thr_pcks in pcks.items():
            wandb.log({f"{split}/pck@{thr}": np.mean(thr_pcks)})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader):
    device = config.get("device", "cuda")
    run_name = config["id"]
    max_steps = config["max_steps"]
    max_epochs = math.ceil(max_steps / len(train_dataloader))

    aggregation_network = aggregation_network.to(device)
    train_end = False
    global_step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(train_dataloader):
            batch = train_helpers.prepare_batch(batch, device)
            imgs, target = batch["source"], batch["heatmap"]
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=config.get("eval_mode", True))
            loss = loss_pose(config, aggregation_network, pred, target)
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
                    validate(config, diffusion_extractor, aggregation_network, val_dataloader, "val", global_step, run_name)
            global_step += 1
            if global_step > max_steps:
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
    train_dataset, train_dataloader = get_mscoco_loader(config, config["train_file"], True)
    val_dataset, val_dataloader = get_mscoco_loader(config, config["val_file"], False)

    config["id"] = run_name
    config["dims"] = diffusion_extractor.dims
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # python3 train_pose.py --config_path configs/train_pose.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)