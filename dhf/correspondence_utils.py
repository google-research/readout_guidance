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
# This code is a subset taken from correspondence_utils.py in Diffusion Hyperfeatures
# and implements helpers for computing nearest neighbor correspondences in feature maps.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/correspondence_utils.py
# =================================================================

import math
import numpy as np
import torch

def rescale_points(points, old_shape, new_shape):
    # Assumes old_shape and new_shape are in the format (w, h)
    # and points are in (y, x) order
    x_scale = new_shape[0] / old_shape[0]
    y_scale = new_shape[1] / old_shape[1]
    rescaled_points = np.multiply(points, np.array([y_scale, x_scale]))
    return rescaled_points

def flatten_feats(feats):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = feats.shape
    feats = feats.view((b, c, -1))
    feats = feats.permute((0, 2, 1))
    return feats

def normalize_feats(feats):
    # (b, w*h, c)
    feats = feats / torch.linalg.norm(feats, dim=-1)[:, :, None]
    return feats

def batch_cosine_sim(img1_feats, img2_feats, flatten=True, normalize=True, low_memory=False):
    if flatten:
        img1_feats = flatten_feats(img1_feats)
        img2_feats = flatten_feats(img2_feats)
    if normalize:
        img1_feats = normalize_feats(img1_feats)
        img2_feats = normalize_feats(img2_feats)
    if low_memory:
        sims = []
        for img1_feat in img1_feats[0]:
            img1_sims = img1_feat @ img2_feats[0].T
            sims.append(img1_sims)
        sims = torch.stack(sims)[None, ...]
    else:
        sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))
    return sims

def find_nn_correspondences(sims):
    """
    Assumes sims is shape (b, w*h, w*h). Returns points1 (w*hx2) which indexes the image1 in column-major order
    and points2 which indexes corresponding points in image2.
    """
    w = h = int(math.sqrt(sims.shape[-1]))
    b = sims.shape[0]
    points1 = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h)), dim=-1)
    points1 = points1.expand((b, w, h, 2))
    # Convert from row-major to column-major order
    points1 = points1.reshape((b, -1, 2))
    
    # Note x = col, y = row
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % h
    points2_y = points2 // h
    points2 = torch.stack([points2_y, points2_x], dim=-1)
    
    points1 = points1.to(torch.float32)
    points2 = points2.to(torch.float32)

    return points1, points2

def find_nn_source_correspondences(img1_feats, img2_feats, source_points, output_size, load_size):
    """
    Precompute nearest neighbor of source_points in img1 to target_points in img2.
    """
    img1_feats = torch.nn.functional.interpolate(img1_feats, load_size, mode="bilinear")
    img2_feats = torch.nn.functional.interpolate(img2_feats, load_size, mode="bilinear")

    source_idx = torch.from_numpy(points_to_idxs(source_points, load_size)).long()
    # Select source_points in the flattened (w, h) dimension as source_idx
    img1_feats = flatten_feats(img1_feats)
    img2_feats = flatten_feats(img2_feats)
    img1_feats = img1_feats[:, source_idx, :]
    img1_feats = normalize_feats(img1_feats)
    img2_feats = normalize_feats(img2_feats)
    sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))

    # Find nn_correspondences but with points1 = source_points
    num_pixels = int(math.sqrt(sims.shape[-1]))
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % num_pixels
    points2_y = points2 // num_pixels
    points2 = torch.stack([points2_y, points2_x], dim=-1)

    points1 = torch.from_numpy(source_points)
    points2 = points2[0]
    return points1, points2

def points_to_idxs(points, load_size):
    points_y = points[:, 0]
    points_y = np.clip(points_y, 0, load_size[1]-1)
    points_x = points[:, 1]
    points_x = np.clip(points_x, 0, load_size[0]-1)
    idx = load_size[1] * np.round(points_y) + np.round(points_x)
    return idx

def points_to_patches(source_points, num_patches, load_size):
    source_points = np.round(source_points)
    new_H = new_W = num_patches
    # Note that load_size is in (w, h) order and source_points is in (y, x) order
    source_patches_y = (new_H / load_size[1]) * source_points[:, 0]
    source_patches_x = (new_W / load_size[0]) * source_points[:, 1]
    source_patches = np.stack([source_patches_y, source_patches_x], axis=-1)
    # Clip patches for cases where it falls close to the boundary
    source_patches = np.clip(source_patches, 0, num_patches - 1)
    source_patches = np.round(source_patches)
    return source_patches

def compute_pck(predicted_points, target_points, load_size, pck_threshold=0.1, target_bounding_box=None):
    distances = np.linalg.norm(predicted_points - target_points, axis=-1)
    if target_bounding_box is None:
        pck = distances <= pck_threshold * max(load_size)
    else:
        left, top, right, bottom = target_bounding_box
        pck = distances <= pck_threshold * max(right-left, bottom-top)
    return distances, pck, pck.sum() / len(pck)