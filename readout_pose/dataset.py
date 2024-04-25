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
import json
import numpy as np
import os
from PIL import Image
import torch

import sys
sys.path.append("../")
from readout_pose import pose_helpers
from readout_training import train_helpers

class MSCOCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_files,
        bucket_root,
        image_root=None,
        keypoint_file=None,
        size=512,
        **kwargs
    ):
        self.bucket_root = bucket_root
        self.image_root = image_root
        self.keypoint_file = keypoint_file
        
        self.size = size
        self.heatmap_size = self.size // 8
        self.heatmap_sigma = self.heatmap_size // 16

        self.split = os.path.basename(self.keypoint_file).split("_")[-1].split(".")[0]
        self.data = json.load(open(f"{self.bucket_root}/{self.keypoint_file}"))["annotations"]
        
        # Filter for samples specified by id
        if len(annotation_files) > 0:
            ids = set(sum([json.load(open(file)) for file in annotation_files], []))
            self.data = [item for item in self.data if item["id"] in ids]
        # Filter for samples with at least one visible keypoint
        self.data = self.filter_keypoints(self.data)
    
    def filter_keypoints(self, data):
        new_data = []
        for item in data:
            keypoints = np.array(item["keypoints"]).reshape(17, 3)
            if (keypoints[..., 2]).sum() > 0:
                new_data.append(item)
        return new_data

    def image_to_array(self, source, source_range):
        source = np.array(source)
        source = einops.rearrange(source, 'w h c -> c w h')
        # Normalize source to [-1, 1]
        source = source.astype(np.float32) / 255.0
        source = train_helpers.renormalize(source, (0, 1), source_range)
        return source

    def preprocess(self, source, tracks, bbox, resize_size):
        crop_x, crop_y, crop_width, crop_height = bbox
        crop_resize_img = lambda img: img.convert("RGB").crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)).resize(resize_size)
        source = crop_resize_img(source)
        tracks[..., 0] = (tracks[..., 0] - crop_x) * resize_size[0] / crop_width
        tracks[..., 1] = (tracks[..., 1] - crop_y) * resize_size[1] / crop_height
        return source, tracks

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = str(item["image_id"]).zfill(12) + ".jpg"
        source = Image.open(os.path.join(self.bucket_root, self.image_root, self.split, image_id))

        meta = pose_helpers.mscoco_to_openpose(item["keypoints"])
        subset = np.array(meta["subset"])
        candidate = np.array(meta["candidate"])

        source, candidate = self.preprocess(source, candidate, item["bbox"], (self.size, self.size))
        source = self.image_to_array(source, (-1, 1))
        heatmap = pose_helpers.meta_to_heatmap(candidate, subset, self.size, self.heatmap_size)
        if self.heatmap_sigma > 0:
            heatmap = np.stack([pose_helpers.heatmap_to_gaussian(h, sigma=self.heatmap_sigma) for h in heatmap])
        
        batch = {
            "source": source,
            "heatmap": heatmap
        }
        return batch