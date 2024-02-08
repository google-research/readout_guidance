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
import torch
from tqdm import tqdm
import wandb
import math

import sys
sys.path.append("../")
from readout_training.dataset import ControlDataset
from readout_training import train_helpers

# ====================
#     Dataloader
# ====================
def get_spatial_loader(config, annotation_file, shuffle):
    dataset = ControlDataset(
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
def loss_spatial(aggregation_network, pred, target, sparse_loss=False, control_range=None):
    target = train_helpers.standardize_feats(pred, target)
    if sparse_loss and control_range is not None:
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        min_value = control_range[0]
        is_zero = target == min_value
        if is_zero.sum() > 0:
            loss = (loss[target == min_value].mean() + loss[target != min_value].mean()) / 2
        else:
            loss = loss.mean()
    else:
        loss = torch.nn.functional.mse_loss(pred, target)
    return loss

# ====================
#  Validate and Train
# ====================
def validate(config, diffusion_extractor, aggregation_network, dataloader, split, step, run_name):
    device = aggregation_network.device
    sparse_loss = config["dataset_args"]["sparse_loss"]
    control_range = config["dataset_args"]["control_range"]
    total_loss = []
    for j, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = train_helpers.prepare_batch(batch, device)
            imgs, target = batch["source"], batch["control"]
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
            loss = loss_spatial(aggregation_network, pred, target, sparse_loss, control_range)
            total_loss.append(loss.item())
            log_max = config.get("log_max")
            if log_max == -1 or j < log_max:
                target = train_helpers.standardize_feats(imgs, target)
                pred = train_helpers.standardize_feats(imgs, pred)
                grid = train_helpers.log_grid(imgs, target, pred, control_range)
                results_folder = train_helpers.make_results_folder(config, split, "preds", run_name=run_name)
                grid.save(f"{results_folder}/step-{step}_b-{j}.png")
            else:
                if split == "train":
                    break
    if split == "val":
        wandb.log({f"{split}/loss": loss})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader):
    device = config.get("device", "cuda")
    run_name = config["id"]
    sparse_loss = config["dataset_args"]["sparse_loss"]
    control_range = config["dataset_args"]["control_range"]
    max_steps = config["max_steps"]
    max_epochs = math.ceil(max_steps / len(train_dataloader))

    aggregation_network = aggregation_network.to(device)
    train_end = False
    global_step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(train_dataloader):
            batch = train_helpers.prepare_batch(batch, device)
            imgs, target = batch["source"], batch["control"]
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = loss_spatial(aggregation_network, pred, target, sparse_loss, control_range)
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
                    validate(config, diffusion_extractor, aggregation_network, train_dataloader, "train", global_step, run_name)
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
    train_dataset, train_dataloader = get_spatial_loader(config, config["train_file"], True)
    val_dataset, val_dataloader = get_spatial_loader(config, config["val_file"], False)

    config["dataset_args"]["sparse_loss"] = train_dataset.sparse_loss
    config["dataset_args"]["control_range"] = train_dataset.control_range
    config["id"] = run_name
    config["dims"] = diffusion_extractor.dims
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # python3 train_spatial.py --config_path configs/train_spatial.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)