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
import glob
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image
from tqdm import tqdm
import torch

import sys
sys.path.append("data/deps/co-tracker")
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

def read_frames(video_folder, max_frames, device):
    frames = list(sorted(glob.glob(f"{video_folder}/*")))
    frames = np.stack([np.array(Image.open(file)) for file in frames])
    # Set a max frame length to avoid OOM
    frames = frames[:min(len(frames), max_frames)]
    frames = torch.from_numpy(frames).to(device)
    frames = frames.permute(0, 3, 1, 2)[None].float()
    return frames

def main(args, device="cuda"):
    config = OmegaConf.load(args.config_path)
    max_frames = config["max_frames"]
    grid_size = config["grid_size"]
    grid_query_frame_interval = config["grid_query_frame_interval"]
    visualize = config["visualize"]
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(device)
    
    for video_folder in tqdm(glob.glob(config["read_root"])):
        try:
            video_name = os.path.basename(video_folder)
            save_folder = f"{config['save_root']}/{video_name}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            frames = read_frames(video_folder, max_frames, device)
            for grid_query_frame in range(0, frames.shape[1], grid_query_frame_interval):
                grid_query_name = str(grid_query_frame).zfill(5)
                tracks, visibles = model(
                    frames,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                    backward_tracking=True,
                )
                if visualize:
                    vis = Visualizer(save_dir=save_folder, pad_value=120, linewidth=3)
                    vis.visualize(frames, tracks, visibles, query_frame=grid_query_frame, filename=grid_query_name)
                tracks = tracks.detach().cpu().numpy()
                visibles = visibles.detach().cpu().numpy()
                np.save(f"{save_folder}/tracks_{grid_query_name}.npy", tracks)
                np.save(f"{save_folder}/visibles_{grid_query_name}.npy", visibles)
                first_frame = frames[:, grid_query_frame][0].permute((1, 2, 0))
                first_frame = first_frame.detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(first_frame).save(f"{save_folder}/images_{grid_query_name}.jpg")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    # conda run -n readout python3 annotate_correspondence.py --config_path configs/annotate_correspondence.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    main(args)