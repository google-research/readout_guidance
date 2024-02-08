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
import einops

import sys
sys.path.append("../")
from readout_training.dataset import TripletDataset
from readout_training import train_helpers

# ====================
#     Dataloader
# ====================
def get_appearance_loader(config, annotation_file, shuffle):
    dataset = TripletDataset(
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
# Triplet loss formulation taken from DreamSim: Learning New Dimensions 
# of Human Visual Similarity using Synthetic Data (Fu et. al. NeurIPS 2023).
def compute_distance(a, b, pool_before_dist=False):
    if pool_before_dist:
        a = a.mean([-2, -1])
        b = b.mean([-2, -1])
        dist = 1 - torch.nn.functional.cosine_similarity(a, b, dim=1)
    else:
        dist = 1 - torch.nn.functional.cosine_similarity(a, b, dim=1)
        dist = dist.mean([-2, -1])
    return dist

def compute_hinge_loss(x, y, margin):
    # https://github.com/ssundaram21/dreamsim/blob/6f4a5182b37a2e255bad9fc471c50be3d8613037/util/train_utils.py#L7
    y_rounded = torch.round(y) # Map [0, 1] -> {0, 1}
    y_transformed = -1 * (1 - 2 * y_rounded) # Map {0, 1} -> {-1, 1}
    return torch.max(torch.zeros(x.shape).to(x.device), margin + (-1 * (x * y_transformed))).sum()

def loss_appearance(aggregation_network, pred, cosine_loss=True, margin=0.5):
    anchor, positive, negative = pred.chunk(3)
    assert cosine_loss is True, "only cosine loss is supported"
    # https://github.com/ssundaram21/dreamsim/blob/6f4a5182b37a2e255bad9fc471c50be3d8613037/dreamsim/model.py#L72C14-L72C65
    # cosine sim higher is better, 1 - cosine sim lower is better
    d_pos = compute_distance(anchor, positive)
    d_neg = compute_distance(anchor, negative)
    # target = 1 means that d1 < d0, d1 is the positive
    target = torch.randint(2, size=d_pos.shape).float().to(d_pos.device)
    # when target == 0 we want d0 to be the positive
    d0 = torch.where(target == 0, d_pos, d_neg)
    # when target == 1 we want d1 to be the positive
    d1 = torch.where(target == 1, d_pos, d_neg)
    decisions = torch.lt(d1, d0)
    logit = d0 - d1
    # Compute a similar logits vs [0, 1] label as LPIPs
    loss = compute_hinge_loss(logit, target, margin)
    acc = (target == 1) == decisions
    acc = acc.float()
    return loss.mean(), acc, d_pos, d_neg

# ====================
#  Validate and Train
# ====================
def validate(config, diffusion_extractor, aggregation_network, dataloader, split, step, run_name):
    device = aggregation_network.device
    log_max = config.get("log_max", None)
    plot_max = config.get("plot_max", None)

    total_loss, total_accuracy = [], []
    for j, batch in tqdm(enumerate(dataloader)):
        with torch.inference_mode():
            if log_max is None or j < log_max:
                batch = train_helpers.prepare_batch(batch, device)
                source, target, control = batch["source"], batch["target"], batch["control"]
                num_items = control.shape[1] + 2
                control = einops.rearrange(control, 'b f c w h -> (b f) c w h')
                imgs = torch.cat([source, target, control], dim=0)
                preds = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
                loss, accuracy, pos_dist, neg_dist = loss_appearance(aggregation_network, preds)
                total_loss.append(loss)
                total_accuracy.append(accuracy)
                if plot_max is None or j < plot_max:
                    batch_size = batch["source"].shape[0]
                    preds = einops.rearrange(preds, '(b f) c w h -> b f c w h', b=batch_size)
                    a = einops.rearrange(preds[:, 0][:, None], 'b f c w h -> (b f) c w h')
                    b = einops.rearrange(preds, 'b f c w h -> (b f) c w h')
                    preds_dist = compute_distance(a, b)
                    txt = train_helpers.log_txt_as_img((imgs.shape[2], imgs.shape[3]), [str(round(dist.item(), 2)) for dist in preds_dist], size=36)
                    imgs = imgs.detach().cpu()
                    imgs = torch.cat([imgs, txt], dim=0)
                    imgs = train_helpers.renormalize(imgs, (-1, 1), (0, 1))
                    grid = train_helpers.make_grid(imgs, num_items)
                    results_folder = train_helpers.make_results_folder(config, split, "preds", run_name=run_name)
                    grid.save(f"{results_folder}/step-{step}_b-{j}.png")
            else:
                break
    total_loss = torch.stack(total_loss)
    total_accuracy = torch.cat(total_accuracy, dim=0)
    wandb.log({f"{split}/loss": total_loss.mean().item()})
    wandb.log({f"{split}/acc": total_accuracy.mean().item()})
    wandb.log({f"{split}/num_samples": total_accuracy.shape[0]})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader):
    device =  aggregation_network.device
    run_name = config["id"]
    max_steps = config["max_steps"]
    accumulation_steps = config.get("accumulation_steps")
    max_epochs = math.ceil(max_steps / len(train_dataloader))

    aggregation_network = aggregation_network.to(device)
    train_end = False
    global_step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(train_dataloader):
            batch = train_helpers.prepare_batch(batch, device)
            source, target, control = batch["source"], batch["target"], batch["control"]
            control = einops.rearrange(control, 'b f c w h -> (b f) c w h')
            imgs = torch.cat([source, target, control], dim=0)
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss, accuracy, pos_dist, neg_dist = loss_appearance(aggregation_network, pred)
            if accumulation_steps:
                # Perform gradient accumulation
                # Normalize by accumulation_steps so 
                # the learning rate is invariant to batch size
                loss = loss / accumulation_steps
                loss.backward()
                if (global_step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Note that we want the pos_dist to be smaller than the neg_dist
            wandb.log({"train/step_loss": loss.mean().item()}, step=global_step)
            wandb.log({"train/pos_dist": pos_dist.mean().item()}, step=global_step)
            wandb.log({"train/neg_dist": neg_dist.mean().item()}, step=global_step)
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
    train_dataset, train_dataloader = get_appearance_loader(config, config["train_file"], True)
    val_dataset, val_dataloader = get_appearance_loader(config, config["val_file"], False)

    config["id"] = run_name
    config["dims"] = diffusion_extractor.dims
    optimizer = train_helpers.load_optimizer(config, diffusion_extractor, aggregation_network)
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # python3 train_appearance.py --config_path configs/train_appearance.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)