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

import numpy as np
import torch
import cv2
import math

import sys
sys.path.append("../")
from readout_guidance import rg_operators

# ========================
#  Pose Format Conversion
# ========================
MSCOCO_TO_OPENPOSE = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

def swap_kv(mapping):
    swap_mapping = {v:i for i, v in enumerate(mapping)}
    return [swap_mapping[v] for v in range(len(swap_mapping))]

def mscoco_to_openpose(keypoints, num_keypoints=17):
    keypoints = np.array(keypoints).reshape(num_keypoints, 3)
    mapping = MSCOCO_TO_OPENPOSE
    keypoints[np.arange(keypoints.shape[0])] = keypoints[mapping]
    # set neck (1) to be the interpolation of both shoulders (2, 5)
    keypoints = np.insert(keypoints, 1, np.zeros((1, 3)), axis=0)
    right_sho, left_sho = keypoints[2], keypoints[5]
    neck = (right_sho[:2] + left_sho[:2]) / 2
    if right_sho[2] != 0 and left_sho[2] != 0:
        keypoints[1] = np.concatenate([neck, np.array([2])])

    visible_idxs = np.where(keypoints[:, 2] != 0)[0]
    num_points = len(visible_idxs)
    keypoints = keypoints[visible_idxs][:, :2]
    candidate_ids = np.arange(num_points)[..., None]
    candidate_scores = np.ones(num_points)[..., None]
    candidate = np.concatenate([keypoints, candidate_scores, candidate_ids], axis=-1)
    
    subset = np.ones(num_keypoints + 1) * -1
    subset[visible_idxs] = np.arange(num_points)
    subset = np.concatenate([subset, np.array([0, num_points])])
    subset = subset[None, ...]
    
    meta = {"candidate": candidate.tolist(), "subset": subset.tolist()}
    return meta

def openpose_to_mscoco(meta, num_keypoints=17, subset_idx=None):
    keypoints = np.zeros((num_keypoints + 1, 2))
    if subset_idx is None:
        # if multiple people, select the one with the most visible keypoints
        num_people = [subset[-1] for subset in meta["subset"]]
        subset_idx = np.argmax(num_people)
    subset = np.array(meta["subset"])[subset_idx]
    subset = subset[:num_keypoints+1]
    candidate = np.array(meta["candidate"])
    visible_idxs = np.where(subset != -1)[0]
    # select the portion of candidate that is present in subset
    keypoints[visible_idxs, :2] = candidate[subset[visible_idxs].astype(np.int32), :2]
    # add visibility
    visibility = np.zeros((num_keypoints + 1, 1))
    visibility[visible_idxs] = 2
    keypoints = np.concatenate([keypoints, visibility], axis=1)
    # remove the neck
    keypoints = np.delete(keypoints, 1, axis=0)
    # remap
    mapping = swap_kv(MSCOCO_TO_OPENPOSE)
    keypoints = keypoints[mapping]
    keypoints = keypoints.reshape(-1).tolist()
    return keypoints

def heatmap_to_gaussian(heatmap, sigma=1.0):
    height, width = heatmap.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    gaussian_map = np.zeros_like(heatmap, dtype=np.float32)
    for y_heatmap, x_heatmap in zip(*np.where(heatmap > 0)):
        distances = np.sqrt((x - x_heatmap)**2 + (y - y_heatmap)**2)
        gaussian_kernel = np.exp(-distances**2 / (2 * sigma**2))
        gaussian_map += gaussian_kernel
    return gaussian_map

def meta_to_heatmap(candidate, subset, size, latent_size, num_keypoints=18):
    assert subset.shape[0] == 1
    in_bounds = lambda p: p >= 0 and p < latent_size
    keypoints = np.zeros((num_keypoints, latent_size, latent_size))
    for i in range(num_keypoints):
        index = int(subset[0][i])
        if index == -1:
            continue
        points = candidate[index][0:2]
        points = rg_operators.rescale_points(points[None, :], (size, size), (latent_size, latent_size))[0]
        x, y = int(points[0]), int(points[1])
        if in_bounds(x) and in_bounds(y):
            keypoints[i, y, x] = 1
    return keypoints

def heatmap_to_meta(heatmap, size, latent_size, num_keypoints=18, thresh=0.1, logits=False):
    heatmap = heatmap.reshape((heatmap.shape[0], heatmap.shape[1], -1))
    if logits:
        heatmap = torch.nn.functional.softmax(heatmap, dim=-1)
    heatmap = heatmap.detach().cpu().numpy()
    batch_candidate, batch_subset = [], []
    for b in range(heatmap.shape[0]):
        # k_ref, x_ref, y_ref = heatmap[b].nonzero()
        idxs = heatmap[b].argmax(axis=-1)
        x = idxs % latent_size
        y = idxs // latent_size
        k = np.arange(idxs.shape[0])
        thresh_idxs = heatmap[b][k, idxs] > thresh
        k, x, y = k[thresh_idxs], x[thresh_idxs], y[thresh_idxs]

        points = np.stack([x, y]).transpose()
        points = rg_operators.rescale_points(points, (latent_size, latent_size), (size, size))
        candidate = np.stack([points[:, 0], points[:, 1], np.ones(x.shape[0]), np.arange(x.shape[0])])
        candidate = candidate.transpose()
        batch_candidate.append(candidate)

        subset = np.ones(num_keypoints) * -1
        subset[k] = np.arange(k.shape[0])
        subset = np.concatenate([subset, np.array([0, k.shape[0]])])
        subset = subset[None, ...]
        batch_subset.append(subset)
    return batch_candidate, batch_subset

def re_center_scale(meta, bbox, size):
    candidate = np.array(meta["candidate"])
    new_candidate = rg_operators.rescale_points(candidate[:, :2], size, bbox[2:])
    new_candidate += bbox[:2]
    new_candidate = np.concatenate([new_candidate, candidate[:, 2:]], axis=-1)
    meta["candidate"] = new_candidate.tolist()
    return meta

def merge_metas(metas):
    # Filter out empty metas
    metas = [meta for meta in metas if len(meta["candidate"]) > 0]
    if len(metas) == 0:
        return {"candidate": [], "subset": []}
    metas = [{
        "candidate": np.array(meta["candidate"]), 
        "subset": np.array(meta["subset"])
    } for meta in metas]
    candidate, subset = metas[0]["candidate"], metas[0]["subset"]
    for i in range(1, len(metas)):
        num_keypoints = candidate.shape[0]
        new_candidate = metas[i]["candidate"]
        new_subset = metas[i]["subset"]
        # Increment index
        new_candidate[..., 3] += num_keypoints
        # Select visibles and omit last two elements
        # (num_keypoints, score)
        visibles = new_subset != -1
        visibles[:, -2:] = False
        new_subset[visibles] += num_keypoints
        # Add new elements
        candidate = np.concatenate([candidate, new_candidate], axis=0)
        subset = np.concatenate([subset, new_subset], axis=0)
    meta = {"candidate": candidate.tolist(), "subset": subset.tolist()}
    return meta

# ====================
#    Pose Rescaling
# ====================
CONNECTIONS = [
    [0, 1], [1, 2], [1, 8], [1, 11], [1, 5],
    [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]
]

def get_joint_to_pos(meta):
    candidate, subset = meta["candidate"], meta["subset"]
    candidate, subset = np.array(candidate), np.array(subset)
    joint_to_pos, joint_to_cand = {}, {}
    assert subset.shape[0] == 1
    # Remove last two indices which are score, num_joints
    for i, id in enumerate(subset[0][:-2]):
        if id == -1:
            continue
        joint_to_cand[i] = int(id)
        joint_to_pos[i] = candidate[joint_to_cand[i]][:2]
    return joint_to_pos, joint_to_cand

def rescale_pair(joint_to_pos, pair, new_length=None, static_joint_to_pos=None):
    if pair[0] not in joint_to_pos or pair[1] not in joint_to_pos:
        return -1
    point1, point2 = joint_to_pos[pair[0]], joint_to_pos[pair[1]]
    if new_length is not None:
        # Use static_joint_to_pos to preserve angles
        static_point1, static_point2 = static_joint_to_pos[pair[0]], static_joint_to_pos[pair[1]]
        direction = (static_point2 - static_point1) / np.linalg.norm(static_point2 - static_point1)
        return point1 + direction * new_length
    else:
        return np.linalg.norm(point2 - point1)

def rescale_meta(ref_meta, meta):
    meta, ref_meta = meta.copy(), ref_meta.copy()
    # Get first person
    meta["subset"] = [meta["subset"][0]]
    ref_meta["subset"] = [ref_meta["subset"][0]]
    # Get positions
    candidate = np.array(meta["candidate"])
    ref_joint_to_pos, _ = get_joint_to_pos(ref_meta)
    static_joint_to_pos, joint_to_cand = get_joint_to_pos(meta)
    joint_to_pos, joint_to_cand = get_joint_to_pos(meta)

    # Get the max joint length in ref
    # Set one limb to have fixed length
    # and everything is rescaled accordingly
    ref_lengths = {i: rescale_pair(ref_joint_to_pos, CONNECTIONS[i]) for i in range(len(CONNECTIONS))}
    pair = CONNECTIONS[max(ref_lengths, key=lambda i: ref_lengths[i])]
    ref_length = rescale_pair(ref_joint_to_pos, pair)
    length = rescale_pair(joint_to_pos, pair)
    
    # Note that CONNECTIONS should be iterated through
    # using breadth first search for this to make sense
    for i, pair in enumerate(CONNECTIONS):
        get_pair_present = lambda pair, joint_to_pos: pair[0] in joint_to_pos and pair[1] in joint_to_pos
        if get_pair_present(pair, joint_to_pos):
            if get_pair_present(pair, ref_joint_to_pos):
                # Rescale point2
                scale = rescale_pair(ref_joint_to_pos, pair) / ref_length
                new_length = length * scale
                new_point = rescale_pair(joint_to_pos, pair, new_length=new_length, static_joint_to_pos=static_joint_to_pos)
                candidate[joint_to_cand[pair[1]]][:2] = new_point
                # Update the joint positions in meta
                meta["candidate"] = candidate
                joint_to_pos, _ = get_joint_to_pos(meta)
            else:
                print("ref_meta must be superset of meta")
    meta["candidate"] = meta["candidate"].tolist()
    return meta

# =============================================================================
# This visualization code is from ControlNet (Zhang et. al., ICCV 2023).
# Original source: https://github.com/lllyasviel/ControlNet/blob/ed85cd1e25a5ed592f7d8178495b4483de0331bf/annotator/openpose/util.py#L37
# =============================================================================
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

# =============================================================================
# This PCK computation code is from mmpose.
# Original source: https://github.com/open-mmlab/mmpose/blob/main/mmpose/evaluation/functional/keypoint_eval.py
# =============================================================================
def _calc_distances(preds, gts, mask, norm_factor):
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T

def _distance_acc(distances, thr=0.5):
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1

def keypoint_pck_accuracy(pred, gt, mask, thr, norm_factor):
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt

# ====================
#   PCK Computation
# ====================
def get_mscoco_joint_locations(keypoints, num_keypoints=17):
    keypoints = np.array(keypoints).reshape((num_keypoints, 3))
    joints = keypoints[:, :2]
    visibles = keypoints[:, 2] != 0
    dist = joints.max(axis=0) - joints.min(axis=0)
    return joints[None, ...], visibles[None, ...], dist[None, ...]

def compute_pose_pck(pred_keypoints, gt_keypoints, thr):
    joints_pred, vis_pred, dist_pred = get_mscoco_joint_locations(pred_keypoints)
    joints_gt, vis_gt, dist_gt = get_mscoco_joint_locations(gt_keypoints)
    if vis_gt.sum() == 0:
        return None
    else:
        acc, avg_acc, cnt = keypoint_pck_accuracy(joints_pred, joints_gt, vis_gt, thr, dist_gt)
        return avg_acc