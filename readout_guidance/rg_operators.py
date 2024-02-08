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
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch

# ======================
#     Common Utils
# ======================
def get_loss_rescale(edit, i):
    loss_rescale = edit.get("loss_rescale", 1)
    if type(loss_rescale) is list and i is not None:
        loss_rescale = loss_rescale[i]
    return loss_rescale

def renormalize(x, range_a, range_b):
    # Note that if any value exceeds 255 in uint8 you get overflow
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def _run_aggregation(feats, aggregation_network, emb=None):
    feats = einops.rearrange(feats, 'b w h c -> b c w h')
    feats = aggregation_network(feats, emb)
    feats = einops.rearrange(feats, 'b c w h -> b w h c')
    return feats

def run_aggregation(obs_feat, gt_feat, aggregation_network=None, emb=None):
    if aggregation_network is not None:
        device = obs_feat.device
        obs_feat = obs_feat.to(aggregation_network.device)
        gt_feat = gt_feat.to(aggregation_network.device)
        obs_feat = _run_aggregation(obs_feat, aggregation_network, emb)
        gt_feat = _run_aggregation(gt_feat, aggregation_network, emb)
        obs_feat, gt_feat = obs_feat.to(device), gt_feat.to(device)
    return obs_feat, gt_feat

# ======================
#  Correspondence Utils
# ======================
def get_cosine_similarity(a, b):
    # Flatten Feats
    a = einops.rearrange(a, 'b h w c -> b (h w) c')
    b = einops.rearrange(b, 'b h w c -> b (h w) c')
    # Normalize Feats
    a = a / torch.linalg.norm(a, dim=-1)[:, :, None]
    b = b / torch.linalg.norm(b, dim=-1)[:, :, None]
    sims = torch.matmul(a, b.permute((0, 2, 1)))
    # Convert from [-1, 1] to [0, 1]
    sims = (sims + 1) / 2
    return sims

def rescale_points(points, old_res, new_res):
    old_h, old_w = old_res
    new_h, new_w = new_res
    points[..., 0] = points[..., 0] * (new_h / old_h)
    points[..., 1] = points[..., 1] * (new_w / old_w)
    return points

def point_to_idx(points, w):
    return points[..., 0] * w + points[..., 1]

def get_valid_points(points, h, w):
    valid_y = torch.logical_and(points[..., 0] >= 0, points[..., 0] < h)
    valid_x = torch.logical_and(points[..., 1] >= 0, points[..., 1] < w)
    valid = torch.logical_and(valid_y, valid_x)
    return valid

def filter_points(points1, points2, h, w):
    valid1 = get_valid_points(points1, h, w)
    valid2 = get_valid_points(points2, h, w)
    valid = torch.logical_and(valid1, valid2)
    points1 = points1[valid]
    points2 = points2[valid]
    return points1, points2

def process_points(edit, h, w):
    points = edit["points"].long()
    points1, points2 = points[0], points[1]
    points1, points2 = filter_points(points1, points2, h, w)
    return points1, points2

# ======================
#        Logging
# ======================

def get_pca(x):
    b, w, h, c = x.shape
    x = x.reshape(-1, x.shape[-1])
    pca = PCA(n_components=3)
    x = pca.fit_transform(x)
    x = einops.rearrange(x, '(b w h) c -> b w h c', b=b, w=w, h=h)
    x_min = x.min(axis=(0, 1))
    x_max = x.max(axis=(0, 1))
    x = (x - x_min) / (x_max - x_min)
    return x
     
def feats_to_rgb(x, latents_scale=None):
    # Only plot if feats has channels=3
    if x.shape[1] != 3:
        return None
    x = x.detach().clone()
    if latents_scale is not None:
        x = renormalize(x, latents_scale, (0, 1))
    x = x.to(torch.float32).cpu().numpy()
    x = einops.rearrange(x, 'b c w h -> b w h c')
    x = x * 255.0
    x = np.clip(x, 0, 255)
    x = x.astype(np.uint8)
    return x

# ======================
#        Losses
# ======================
def loss_correspondence(obs_feat, gt_feat, edit):
    sims = get_cosine_similarity(gt_feat, obs_feat)
    _, h, w, _ = gt_feat.shape
    points1, points2 = process_points(edit, h, w)
    points1 = point_to_idx(points1, w)
    points2 = point_to_idx(points2, w)
    loss = -sims[:, points1, points2].mean()
    # Add anti-ghosting loss that discourages identity
    # correspondences, i.e., it encourages some kind of motion
    loss = loss * 0.5 + sims[:, points1, points1].mean() * 0.5
    obs_feat = einops.rearrange(obs_feat, 'b w h c -> b c w h')
    gt_feat = einops.rearrange(gt_feat, 'b w h c -> b c w h')
    return loss, obs_feat, gt_feat

def loss_spatial(obs_feat, gt_feat, edit, latents_scale=(0, 1)):
    obs_feat = einops.rearrange(obs_feat, 'b w h c -> b c w h')
    gt_feat = einops.rearrange(gt_feat, 'b w h c -> b c w h')
    control_feat = edit["control"][None, ...].detach().clone()
    control_feat = control_feat.to(obs_feat.dtype)
    _, _, obs_w, obs_h = obs_feat.shape
    # Normalize from arbitrary output scale
    obs_feat = renormalize(obs_feat, edit["control_range"], latents_scale)
    gt_feat = renormalize(gt_feat, edit["control_range"], latents_scale)
    control_feat = renormalize(control_feat, edit["control_range"], latents_scale)
    control_feat = torch.nn.functional.interpolate(control_feat, (obs_w, obs_h))
    loss = torch.nn.functional.mse_loss(obs_feat, control_feat, reduction="none")
    loss = loss.mean()
    return loss, obs_feat, gt_feat

def loss_appearance(obs_feat, gt_feat, edit, latents_scale=(0, 1)):
    gt_feat = einops.rearrange(gt_feat, 'b w h c -> b c w h')
    obs_feat = einops.rearrange(obs_feat, 'b w h c -> b c w h')
    if edit.get("pool", False):
        obs_feat = obs_feat.mean(dim=[-2, -1])
    if edit.get("bbox") is not None:
        # Constraint the appearance similarity loss
        # within some pre-defined bounding box
        gt_bbox = edit["bbox"][0]
        obs_bbox = edit["bbox"][1]
        _, _, w, h = obs_feat.shape
        obs_feat = obs_feat[:, :, obs_bbox[1]:obs_bbox[3], obs_bbox[0]:obs_bbox[2]]
        gt_feat = gt_feat[:, :, gt_bbox[1]:gt_bbox[3], gt_bbox[0]:gt_bbox[2]]
        obs_feat = torch.nn.functional.interpolate(obs_feat, size=(w, h), mode="bilinear")
        gt_feat = torch.nn.functional.interpolate(gt_feat, size=(w, h), mode="bilinear")
    loss_spatial = 1 - torch.nn.functional.cosine_similarity(obs_feat, gt_feat, dim=1)
    loss = loss_spatial.mean()
    loss = loss * (latents_scale[1] - latents_scale[0]) ** 2
    return loss, obs_feat, gt_feat

def loss_guidance(controller, feats, batch_idx, gt_idx, edits=[], log=False, emb=None, latents_scale=None, t=None, i=None):
    rg_loss = 0
    for edit in edits:
        aggregation_network = edit.get("aggregation_network", None)
        gt_feat = feats[gt_idx][None, ...].detach().clone()
        obs_feat = feats[batch_idx][None, ...]
        # Run aggregation network with the option to
        # deactive the timestep embedding conditioning (use_emb)
        obs_feat, gt_feat = run_aggregation(
            obs_feat, 
            gt_feat, 
            aggregation_network, 
            emb if edit.get("use_emb", True) else None
        )
        if edit["head_type"] == "spatial":
            spa_loss, obs_feat, gt_feat = loss_spatial(obs_feat, gt_feat, edit, latents_scale)
            rg_loss += spa_loss * get_loss_rescale(edit, i)
        elif edit["head_type"] == "appearance":
            app_loss, obs_feat, gt_feat = loss_appearance(obs_feat, gt_feat, edit, latents_scale)
            rg_loss += app_loss * get_loss_rescale(edit, i)
        elif edit["head_type"] == "correspondence":
            corr_loss, obs_feat, gt_feat = loss_correspondence(obs_feat, gt_feat, edit)
            rg_loss += corr_loss * get_loss_rescale(edit, i)
        if log:
            feat = feats_to_rgb(torch.concatenate([obs_feat, gt_feat], dim=0), latents_scale)
            if feat is not None:
                controller.obs_feat = Image.fromarray(feat[0])
                controller.gt_feat = Image.fromarray(feat[1])
            else:
                controller.obs_feat = None
                controller.gt_feat = None
    return rg_loss