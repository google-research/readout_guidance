# Readout Head Training
Here, we provide the code for training custom readout heads on your own data. You can then try out these heads by replacing `rg_kwargs.aggregation_kwargs.aggregation_ckpt` in the appropriate config file when running a Readout Guidance demo.

## Pseudo Labels
Before you start training, you need to prepare your data. We also provide helpers for structuring your json files used as annotations in `demo_train.ipynb`.

### Code Dependencies
To automatically download the following repositories run `./data/scripts/download_deps.sh`.
- [ControlNet](https://github.com/lllyasviel/ControlNet) [1]: Includes spatial annotators with OpenPose, MiDaS, HED for image inputs.
- [CoTracker](https://github.com/facebookresearch/co-tracker) [2]: Point tracker to predict correspondences for video inputs.
- [SDEdit](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py) [3]: Implementation from diffusers (also called StableDiffusionImg2ImgPipeline) to globally edit images according to a text prompt.

### Raw Data
To automatically download the following datasets `./data/scripts/download_raw.sh`. We also provide prompts pseudo labeled by BLIP [4] in the annotation files contained in `annotations`.
- [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) [5]: Dataset of 17k images. We use the PascalVOC 2012 images.
- [DAVIS](https://davischallenge.org/davis2017/code.html) [6]: Dataset of 90 videos. We use the DAVIS 2017 semi-supervised 480p videos.

### Annotation Scripts
```
# Annotate with ControlNet spatial annotators
conda run -n control python3 annotate_spatial.py --config_path configs/annotate_spatial.yaml

# Annotate with CoTracker
conda run -n readout python3 annotate_correspondence.py --config_path configs/annotate_correspondence.yaml

# Annotate with SDEdit
conda run -n readout python3 annotate_appearance.py --config_path configs/annotate_appearance.yaml
```

## Training Scripts
```
# Train pose head
python3 train_spatial.py --config_path configs/train_spatial.yaml

# Train correspondence feature head
python3 train_correspondence.py --config_path configs/train_correspondence.yaml

# Train appearance similarity head
python3 train_appearance.py --config_path configs/train_appearance.yaml
```

## Citations
[1] Zhang et. al. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023.\
[2] Karaev et. al. CoTracker: It is Better to Track Together. arXiv 2023.\
[3] Meng et. al. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. ICLR 2022.\
[4] Li et. al. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML 2022.\
[5] Everingham et. al. The PASCAL Visual Object Classes
Challenge 2012 (VOC2012) Results.\
[6] Pont Tuset et. al. The 2017 DAVIS challenge on video object segmentation.