# Readout Head for Pose Estimation

The pose head described in our paper represented the outputs as RGB images: Instead of directly predicting keypoints, the head predicts a *visualization* of the skeleton. Although this approach is useful for quickly training readout heads for guidance, it's not useful for the actual task of pose estimation.

Here, we train **new pose keypoint heads** to answer the following question: **How good are readout heads at pose estimation?** We find that this readout head for SDXL performs similarly to ControlNet's default OpenPose annotator, while the readout head for SDv1-5 actually outperforms it.

## Approach
Here, we outline the design of the pose keypoint head.

#### 1. Training Set
We train on MSCOCO images and their ground-truth human-annotated keypoints. To ensure that the input contains a single person, we crop the image to the ground-truth bounding box corresponding to the ground-truth keypoints. At inference time, the head can operate on full images with a single person, or an image cropped to the person using an off-the-shelf person detector, similar to ViTPose [3].

#### 2. Output Format
We parameterize the readout head to predict `k` heatmaps, where `k` is the number of keypoints. We use `k=18`, corresponding to the first 18 keypoints in the the [OpenPose Body_25 format](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html#pose-output-format-body_25), the same format used for our original RGB pose head.

During training, we follow the same setup as HRNet [1]. We create the ground-truth heatmaps by applying a 2D Gaussian centered at the ground-truth keypoint location with a standard deviation that is `1/16` the size of the heatmap. For example, for a heatmap of size `64x64` we use a Gaussian with standard deviation `4`. We train with a MSE loss between the predicted and ground-truth heatmap.

During inference, we predict the keypoint location by taking the argmax of the corresponding heatmap and its visibility based on whether the value is above or below some pre-determined threshold. We use the threshold `t=0.1`.

#### 3. Readout Head Architecture
We use the same architecture as the original RGB pose head, except we set `output_head_channels=18` and we remove the final tanh activation by setting `output_head_act=False`. For simplicity, we only train this head on a single timestep by setting `eval_mode=True` during training, which sets the inputs to be clean images at the cleanest timestep.

## Results
We report the performance of the pose keypoint head on the unseen evaluation set MSCOCO val2017. For all methods, we crop the input image to the ground-truth bounding box for each person and resize it to a fixed resolution. When computing the metric, we re-scale and re-center the keypoints according to this bounding box to the original image coordinates. We filter out all ground-truth annotations with no visible keypoints. We report the percentage of correct keypoints (PCK).

Our readout heads are trained on MSCOCO train2017 with a learning rate of 1e-3 and batch size of 8. We report the performance of the readout head with `max_steps` equal to 5k or 100k, where the former setting benchmarks the performance when training for at most three hours. Our released weights correspond to the readout head with longer training.

**Table 1. MSCOCO val2017**
| Method | Input Resolution | PCK @ 0.05 | PCK @ 0.1 | PCK @ 0.2 |
|----------|----------|----------|----------|----------|
| **Baselines** |
| [OpenPose](https://github.com/lllyasviel/ControlNet/tree/main/annotator/openpose) [2] | 512x512 | 69.0 | 77.1 | 83.0 | 24.5 |
| [ViTPose](https://github.com/JunkyByte/easy_ViTPose) [3] |  512x512 | 83.4 | 91.9 | 96.1 |
| **SDv1-5** |
| RG (5k steps) | 512x512 | 68.1| 78.8 | 86.2 |
| [RG (100k steps)](https://huggingface.co/g-luo/readout-guidance/resolve/main/readout_pose/readout_sdv15_pose_keypoint.pt?download=true) | 512x512 | 74.0 | 83.8 | 89.7 |
| **SDXL** |
| RG (5k steps) | 1024x1024 | 55.9 | 70.0 | 82.5 |
| [RG (100k steps)](https://huggingface.co/g-luo/readout-guidance/resolve/main/readout_pose/readout_sdxl_pose_keypoint.pt?download=true) | 1024x1024 | 65.9 | 77.5 | 85.6 |

## Code
## Setup
To run the following code, make the following updates to the `readout` conda environment.
```
conda activate readout
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics
```

#### Model Weights
From our HuggingFace page, download the pre-trained weights of the [pose keypoint heads](https://huggingface.co/g-luo/readout-guidance/readout_pose) and move them to the path `readout-guidance/readout_pose/weights`.

#### Raw Data
We use the [MSCOCO](https://cocodataset.org/#download) train2017 and val2017 images and annotations. To automatically download the dataset run `./data/scripts/download_raw.sh`.

#### Training Script
```
python3 train_pose.py --config_path configs/train_pose.yaml
```

#### Annotation Script
```
python3 annotate_pose.py --config_path configs/annotate_pose.yaml
```

## Citations
[1] Wang et. al. Deep High-Resolution Representation Learning for Visual Recognition. TPAMI 2019.\
[2] Cao et. al. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. TPAMI 2019.\
[3] Xu et. al. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. NeurIPS 2022.
