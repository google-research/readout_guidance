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
import glob
import json
import os
import numpy as np
from PIL import Image
import torch

from readout_training import train_helpers

class ControlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_files,
        bucket_root,
        control_root=None,
        image_root=None,
        annotation_max=-1,
        size=512,
        shuffle_once=True,
        # control hparams
        control_type=None,
        sparse_loss=False,
        control_range=(-1.0, 1.0),
    ):
        super().__init__()

        self.bucket_root = bucket_root
        self.control_root = control_root
        self.image_root = image_root
        self.size = size

        # If control_type is defined, use preset hparams
        # depth uses sparse_loss False, control_range (-1.0, 1.0)
        if control_type in ["pose", "edge"]:
            self.sparse_loss = True
            self.control_range = (-0.5, 0.5)
        else:
            self.sparse_loss = sparse_loss
            self.control_range = control_range

        # Shuffle annotations
        self.data = []
        for file in annotation_files:
            self.data.extend(json.load(open(file)))
        if shuffle_once:
            self.data = np.random.permutation(self.data).tolist()
        if annotation_max > 0:
            self.data = self.data[:annotation_max]
        
        # Filter for only samples where the pseudo label exists
        if control_root:
            self.data = train_helpers.filter_anns(self.data, f"{bucket_root}/{control_root}/*")
        
        print(f"Using control_range {control_range}")

    def preprocess(self, source, target, control=None, tracks=None, mask=None, resize_size=(512, 512)):
        width, height = source.size
        crop_size = min(source.size)
        crop_x = np.random.randint(0, width - crop_size + 1)
        crop_y = np.random.randint(0, height - crop_size + 1)
        crop_resize_img = lambda img: img.convert("RGB").crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)).resize(resize_size)
        if source is not None:
            source = crop_resize_img(source)
        if target is not None:
            target = crop_resize_img(target)
        if control is not None:
            control = crop_resize_img(control)
        if tracks is not None:
            # Crop tracks like how image is cropped
            tracks[..., 0] = (tracks[..., 0] - crop_x) * resize_size[0] / crop_size
            tracks[..., 1] = (tracks[..., 1] - crop_y) * resize_size[1] / crop_size
        if mask is not None:
            mask = crop_resize_img(mask)
        return source, target, tracks, control, mask
        
    def image_to_array(self, source, source_range):
        source = np.array(source)
        source = einops.rearrange(source, 'w h c -> c w h')
        # Normalize source to [-1, 1]
        source = source.astype(np.float32) / 255.0
        source = train_helpers.renormalize(source, (0, 1), source_range)
        return source

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Load control
        source = Image.open(os.path.join(self.bucket_root, self.image_root, item["source"]))
        control = Image.open(os.path.join(self.bucket_root, self.control_root, item["source"]))
        target = None
        tracks = None
        mask = None

        source, target, tracks, control, mask = self.preprocess(source, target, tracks=tracks, control=control, mask=mask, resize_size=(self.size, self.size))
        source = self.image_to_array(source, (-1, 1))
        control = self.image_to_array(control, self.control_range)
        
        batch = {
            "source": source,
            "control": control
        }
        return batch

class TripletDataset(ControlDataset):
    def __init__(
        self, 
        annotation_files,
        bucket_root,
        sdedit_root=None,
        image_root=None,
        annotation_max=-1,
        size=512,
        shuffle_once=True,
        min_dist=-1, 
        max_dist=-1,
        # sdedit hparams
        sdedit_select=None,
        num_samples=2,
        target_select="aligned",
        objectives=["rs"]
    ):
        super().__init__(
            annotation_files, 
            bucket_root=bucket_root, 
            image_root=image_root,
            annotation_max=annotation_max, 
            size=size, 
            shuffle_once=shuffle_once
        )
        self.sdedit_root = sdedit_root
        self.sdedit_select = sdedit_select
        self.num_samples = num_samples
        self.target_select = target_select
        self.objectives = objectives

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.data = train_helpers.filter_dist(self.data, min_dist, max_dist)

        # Filter for only samples where the pseudo label exists
        if sdedit_root:
            self.data = train_helpers.filter_anns(self.data, f"{bucket_root}/{sdedit_root}/{sdedit_select}/*/*")

    def path_to_frame(self, f):
        return int(os.path.basename(f).split(".")[0])

    def filter_paths(self, root_path, filter_fn):
        if type(root_path) is not list:
            frame_paths = [path for path in glob.glob(f"{os.path.dirname(root_path)}/*") if filter_fn(path)]
        else:
            frame_paths = [path for path in root_path if filter_fn(path)]
        return frame_paths
    
    def preprocess_img(self, img, crop_x, crop_y, crop_size, resize_size):
        img = img.convert("RGB").crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)).resize(resize_size)
        img = self.image_to_array(img, (-1, 1))
        return img

    def compare_with_dist(self, paths_a, paths_b, source_path):
        paths_a = np.random.permutation(paths_a)
        paths_b = np.random.permutation(paths_b)
        # Ensure that paths_a and paths_b never overlap
        paths_b = paths_b[:self.num_samples-1]
        paths_a = [path for path in paths_a if path not in paths_b]
        dist_a = [np.abs(self.path_to_frame(path) - self.path_to_frame(source_path)) for path in paths_a]
        dist_b = [np.abs(self.path_to_frame(path) - self.path_to_frame(source_path)) for path in paths_b]
        if "max" in self.target_select:
            paths_a = [path for i, path in enumerate(paths_a) if dist_a[i] > max(dist_b)]
        if len(paths_a) > 0 and len(paths_b) == self.num_samples-1:
            return paths_a[0], paths_b
        else:
            return None, None

    def compare_with_other(self, paths_a, paths_b):
        paths_a = np.random.permutation(paths_a)
        paths_b = np.random.permutation(paths_b)
        return paths_a[0], paths_b[:self.num_samples-1]

    def __getitem__(self, idx, return_meta=False):
        item = self.data[idx]
        source_path = os.path.join(self.bucket_root, self.image_root, item["source"])
        source_paths = self.filter_paths(source_path, lambda path: path != source_path)
        sdedit_paths = os.path.join(self.bucket_root, self.sdedit_root, self.sdedit_select, item["source"])
        if "aligned" in self.target_select:
            sdedit_paths = self.filter_paths(sdedit_paths, lambda path: item["source"] in path)
        other_path =  os.path.join(self.bucket_root, self.image_root, "*/*")
        other_paths = self.filter_paths(other_path, lambda path: path.split("/")[-2] != source_path.split("/")[-2])

        source = Image.open(source_path)
        width, height = source.size
        preprocess_kwargs = {
            "resize_size":  (self.size, self.size),
            "crop_size": min(source.size),
            "crop_x": np.random.randint(0, width - min(source.size) + 1),
            "crop_y": np.random.randint(0, height - min(source.size) + 1)
        }
        source = self.preprocess_img(source, **preprocess_kwargs)
        
        # By default the objective should be set only to ["rs"]
        objective = np.random.permutation(self.objectives)[0]
        if objective == "ro":
            # Case 1. (real, other) = sensitive to color
            target_path, control_paths = self.compare_with_other(source_paths, other_paths)
        elif objective == "so":
            # Case 2. (sdedit, other) = sensitive to color
            target_path, control_paths = self.compare_with_other(sdedit_paths, other_paths)
        elif objective == "rr":
            # Case 3. (real far, real close) = agnostic to structure
            target_path, control_paths = self.compare_with_dist(source_paths, source_paths, source_path)
        elif objective == "rs":
            # Case 4. (real far, sdedit close) = sensitive to identity
            target_path, control_paths = self.compare_with_dist(source_paths, sdedit_paths, source_path)

        if target_path is None or control_paths is None:
            # Run Case 1 as a backup
            target_path, control_paths = self.compare_with_other(source_paths, other_paths)
        
        if "matched" in self.target_select:
            # Make target_path and control_path aligned
            target_path = "/".join(target_path.split("/")[:-2])
            target_path += "/" + "/".join(control_paths[0].split("/")[-2:])

        target = Image.open(target_path)
        target = self.preprocess_img(target, **preprocess_kwargs)
        controls = [Image.open(control_path) for control_path in control_paths]
        controls = [self.preprocess_img(control, **preprocess_kwargs) for control in controls]
        control = np.stack(controls)
        batch = {
            "source": source,
            "target": target,
            "control": control
        }

        meta = {"target_path": target_path, "control_path": control_paths, "source_path": source_path, "preprocess_kwargs": preprocess_kwargs}
        if return_meta:
            return batch, meta
        else:
            return batch

class PointsDataset(ControlDataset):
    def __init__(
        self, 
        annotation_files,
        bucket_root,
        points_root=None,
        mask_root=None,
        image_root=None,
        annotation_max=-1,
        size=512,
        shuffle_once=True,
        min_dist=-1,
        max_dist=-1,
        max_frame_idx=128,
        num_points=None
    ):
        super().__init__(
            annotation_files,
            bucket_root=bucket_root,
            image_root=image_root,
            annotation_max=annotation_max, 
            size=size, 
            shuffle_once=shuffle_once
        )
        self.points_root = points_root
        self.mask_root = mask_root
        self.max_frame_idx = max_frame_idx
        self.num_points = num_points

        self.data = train_helpers.filter_dist(self.data, min_dist, max_dist)
        # Filter for max frame index due to point tracking
        self.data = [item for item in self.data if max(train_helpers.get_frame_idx(item["source"]), train_helpers.get_frame_idx(item["target"])) < max_frame_idx]
        # Filter for only samples where the pseudo label exists
        if points_root:
            self.data = train_helpers.filter_anns(self.data, f"{bucket_root}/{points_root}/*", "video_name")

    def __getitem__(self, idx):
        item = self.data[idx]
        source = Image.open(os.path.join(self.bucket_root, self.image_root, item["source"]))
        target = Image.open(os.path.join(self.bucket_root, self.image_root, item["target"]))
        i, j = train_helpers.get_frame_idx(item["source"]), train_helpers.get_frame_idx(item["target"])

        # Load point tracks
        video_name = item["source"].split("/")[0]
        tracks, visibles = train_helpers.open_points(i, j, video_name, self.max_frame_idx, self.bucket_root, self.points_root)
        mask = train_helpers.open_mask(i, video_name, self.bucket_root, self.mask_root)

        source, target, tracks, control, mask = self.preprocess(source, target, tracks=tracks, mask=mask, resize_size=(self.size, self.size))
        source, target = np.array(source), np.array(target)
        source = einops.rearrange(source, 'w h c -> c w h')
        target = einops.rearrange(target, 'w h c -> c w h')
        source = (source.astype(np.float32) / 127.5) - 1.0 # Normalize source images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0 # Normalize target images to [-1, 1].
        
        batch = {
            "source": source,
            "target": target
        }
        
        _, width, height = source.shape
        tracks = train_helpers.filter_tracks(tracks, width, height, mask, self.num_points)
        if tracks is not None:
            # Normalize the scale to [0, 1]
            # points_x should be (0, width - 1)
            tracks[..., 0] = tracks[..., 0] / (width - 1)
            tracks[..., 1] = tracks[..., 1] / (height - 1)
            tracks = np.clip(tracks, 0, 1)
        batch["points"] = tracks

        return batch