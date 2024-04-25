import argparse
import einops
import glob
import json
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
import torch
from tqdm import tqdm

import sys
sys.path.append("../")
from readout_pose import train_pose
from readout_training import train_helpers

def image_to_array(source, source_range):
    source = np.array(source)
    source = einops.rearrange(source, 'w h c -> c w h')
    # Normalize source to [-1, 1]
    source = source.astype(np.float32) / 255.0
    source = train_helpers.renormalize(source, (0, 1), source_range)
    return source

def create_batch(paths, size, device):
    imgs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize(size)
        image = image_to_array(image,  (-1, 1))
        image = torch.from_numpy(image)
        imgs.append(image)
    imgs = torch.stack(imgs)
    imgs = imgs.to(device)
    return {"source": imgs}

def filter_paths(paths, filter_file=None):
    if filter_file is not None:
        filter_names = set(json.load(open(filter_file)))
        paths = [image for image in paths if int(os.path.basename(image).split(".")[0]) in filter_names]
        if len(paths) == 0:
            message = """
            WARNING! After filtering there are 0 paths.
            Make sure to remove filter_file from the config if you 
            would not like to filter any of the paths.
            """
            print(message)
    return paths

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def main(args):
    config = OmegaConf.load(args.config_path)
    paths = glob.glob(config["read_root"])
    paths = filter_paths(paths, config.get("filter_file"))
    
    # Create folders
    save_folder = config["save_root"]
    make_folder(save_folder)
    make_folder(f"{save_folder}_meta")

    # Init models
    device = config.get("device", "cuda")
    batch_size = config["batch_size"]
    size = (config["res"], config["res"])
    _, diffusion_extractor, aggregation_network = train_helpers.load_models(ckpt_path=config["aggregation_ckpt"])
    aggregation_network = aggregation_network.to(device)

    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i+batch_size]
        batch = create_batch(batch_paths, size, device)

        with torch.no_grad():
            imgs = batch["source"]
            pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
            pred_meta = train_pose.heatmap_to_meta(imgs, pred)

        for j in range(batch_size):
            name = os.path.basename(batch_paths[j])
            name = name.split('.')[0]
            meta = pred_meta[j]

            control = train_pose.draw_pose(size, meta)
            control.save(f"{save_folder}/{name}.png")

            meta["candidate"] = meta["candidate"].tolist()
            meta["subset"] = meta["subset"].tolist()
            json.dump(meta, open(f"{save_folder}_meta/{name}.json", "w"))

if __name__ == "__main__":
    # python3 annotate_pose.py --config_path configs/annotate_pose.yaml
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	main(args)