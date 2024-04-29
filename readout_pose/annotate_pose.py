import argparse
import einops
import glob
import json
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
import torch
from torchvision.transforms.functional import crop
from tqdm import tqdm

import sys
sys.path.append("../")
from readout_pose import train_pose, pose_helpers
from readout_training import train_helpers
from readout_guidance import rg_operators

# ====================
#    Person Detector
# ====================
def xyxy_xywh(bbox):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    bbox = torch.cat([bbox[:, :2], w[:, None], h[:, None]], dim=-1)
    return bbox

def get_bboxes(detector, imgs, classes=[0], thresh=0.35):
    # https://github.com/JunkyByte/easy_ViTPose/blob/f1eb4a6147aae5f3a5363d948636fd2e8335acf8/easy_ViTPose/inference.py#L236
    # https://docs.ultralytics.com/modes/predict/#inference-sources
    imgs = train_helpers.renormalize(imgs, (-1, 1), (0, 1))
    results = detector(imgs, classes=classes)
    batch_bbox = []
    for result in results:
        bbox = result.boxes.data
        visibles = bbox[:, 4] > thresh
        bbox = bbox[:, :4][visibles]
        bbox = xyxy_xywh(bbox)
        batch_bbox.append(bbox)
    return batch_bbox

def crop_imgs(imgs, bboxes, size):
    crop_imgs = []
    for i in range(imgs.shape[0]):
        for bbox in bboxes[i]:
            kwargs = ["left", "top", "width", "height"]
            kwargs = {kwargs[j]: int(bbox[j]) for j in range(len(bbox))}
            crop_img = crop(imgs[i][None, ...], **kwargs)
            crop_img = torch.nn.functional.interpolate(crop_img, size)
            crop_imgs.append(crop_img)
    crop_imgs = torch.vstack(crop_imgs)
    return crop_imgs

def get_pose(diffusion_extractor, aggregation_network, imgs, size, eval_mode=True, bboxes=None):
    if bboxes is None:
        pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, imgs, eval_mode=True)
        pred_meta = train_pose.heatmap_to_meta(imgs, pred)
    else:
        # Subfilter imgs to those with bboxes
        batch_size = imgs.shape[0]
        empty_meta = {"candidate": [], "subset": []}
        pred_meta = [empty_meta for _ in range(batch_size)]
        bboxes_visible = [i for i in range(batch_size) if bboxes[i].shape[0] > 0]
        if len(bboxes_visible) > 0:
            bboxes = [bboxes[i] for i in bboxes_visible]
            imgs = imgs[bboxes_visible]
            bboxes_flat = torch.vstack(bboxes)
            bbox_imgs = crop_imgs(imgs, bboxes, size)
            # Run inference
            pred_meta_flat = []
            for i in range(0, bbox_imgs.shape[0], batch_size):
                bbox_pred = train_helpers.get_hyperfeats(diffusion_extractor, aggregation_network, bbox_imgs[i:i+batch_size], eval_mode=True)
                bbox_pred_meta = train_pose.heatmap_to_meta(bbox_imgs[i:i+batch_size], bbox_pred)
                bbox_pred_meta = [pose_helpers.re_center_scale(bbox_pred_meta[j], bboxes_flat[len(pred_meta_flat) + j].tolist(), size) for j in range(len(bbox_pred_meta))]
                pred_meta_flat.extend(bbox_pred_meta)
            # Group multiple people to single image
            for i, visible_idx in enumerate(bboxes_visible):
                start = sum([bbox.shape[0] for bbox in bboxes[:i]])
                end = sum([bbox.shape[0] for bbox in bboxes[:i+1]])
                meta = pose_helpers.merge_metas(pred_meta_flat[start:end])
                pred_meta[visible_idx] = meta
    pred_meta = [{k: v.tolist() if type(v) is np.ndarray else v for k, v in meta.items()} for meta in pred_meta]
    return pred_meta

# ====================
#      Dataloader
# ====================
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

# ====================
#        Eval
# ====================
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

    if "detector_ckpt" in config:
        from ultralytics import YOLO
        detector = YOLO(config["detector_ckpt"], task='detect')
    else:
        detector = None

    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i+batch_size]
        batch = create_batch(batch_paths, size, device)

        with torch.no_grad():
            imgs = batch["source"]
            if detector is not None:
                bboxes = get_bboxes(detector, imgs)
            else:
                bboxes = None
            pred_meta = get_pose(diffusion_extractor, aggregation_network, imgs, size, eval_mode=True, bboxes=bboxes)

        for j in range(batch_size):
            name = os.path.basename(batch_paths[j])
            name = name.split('.')[0]
            meta = pred_meta[j]

            control = train_pose.draw_pose(size, meta)
            control.save(f"{save_folder}/{name}.png")

            meta["candidate"] = meta["candidate"]
            meta["subset"] = meta["subset"]
            json.dump(meta, open(f"{save_folder}_meta/{name}.json", "w"))

if __name__ == "__main__":
    # python3 annotate_pose.py --config_path configs/annotate_pose.yaml
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	main(args)