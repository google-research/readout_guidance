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
import json
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("data/deps/ControlNet")
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.hed import HEDdetector
from annotator.util import resize_image, HWC3

def main(args):
    config = OmegaConf.load(args.config_path)
    mode = config["mode"]
    save_meta = config.get("save_meta", False)
    filter_file = config.get("filter_file", None)
    if filter_file:
        filter_names = set(json.load(open(filter_file)))
    else:
        filter_names = None

    model_cls = {
        "midas": MidasDetector,
        "openpose": OpenposeDetector,
        "hed":HEDdetector,
    }
    model = model_cls[mode]()

    for image in tqdm(glob.glob(config["read_root"])):
        if filter_names and os.path.basename(image) not in filter_names:
            continue
        img = Image.open(image)
        original_res = img.size
        img = np.array(img)
        img = resize_image(HWC3(img), config["res"])
        # Predict pseudo label
        meta = None
        if mode in ["midas", "openpose"]:
            control, meta = model(img)
        elif mode in ["hed"]:
            control = model(img)
        control = Image.fromarray(control)
        control = control.resize(original_res)
        # Save pseudo label
        save_folder = config["save_root"]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        control.save(f"{save_folder}/{os.path.basename(image)}")
        # Optionally save metadata
        if save_meta and meta is not None:
            if not os.path.exists(f"{save_folder}_meta"):
                os.makedirs(f"{save_folder}_meta", exist_ok=True)
            json.dump(meta, open(f"{save_folder}_meta/{os.path.basename(image).split('.')[0]}.json", "w"))

if __name__ == "__main__":
    # conda run -n control python3 annotate_spatial.py --config_path configs/annotate_spatial.yaml
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	main(args)