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
# This code is adapted from aggregation_network.py in Diffusion Hyperfeatures
# and implements a learned aggregation of diffusion features, 
# with additional functionality for feeding the features to an output_head.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/aggregation_network.py
# =================================================================

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        num_norm_groups = kwargs.pop("num_norm_groups")
       
        super().__init__(*args, **kwargs)
        self.norm = nn.GroupNorm(num_norm_groups, kwargs["out_channels"])

    def forward(self, x):
        x = F.conv2d(
            x, 
            self.weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )
        x = self.norm(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_norm_groups=32,
        emb_channels=1280
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                num_norm_groups=num_norm_groups
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            num_norm_groups=num_norm_groups
        )
        self.conv2 = Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=1,
            dilation=1,
            num_norm_groups=num_norm_groups
        )
        self.conv3 = Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            num_norm_groups=num_norm_groups
        )

        # Weight initialization
        if self.shortcut is not None:
            weight_init.c2_msra_fill(self.shortcut)
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

        # Create timestep conditioning layers
        if emb_channels > 0:
            self.emb_layers = nn.Linear(emb_channels, bottleneck_channels)
        else:
            self.emb_layers = nn.Identity()

    def forward(self, x, emb=None):
        out = self.conv1(x)
        out = F.relu(out)

        # Add timestep conditioning
        if emb is not None:
            emb = emb.to(out.dtype)
            emb = emb.to(out.device)
            if emb.shape[0] > out.shape[0]:
                emb = emb[:out.shape[0]]
            emb_out = self.emb_layers(emb)
            emb_out = F.relu(emb_out)
            out = out + emb_out[:, :, None, None]

        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu(out)
        return out

class AggregationNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384,
            num_norm_groups=32,
            save_timestep=[],
            num_timesteps=None,
            use_output_head=False,
            output_head_channels=3,
            output_head_act=True,
            bottleneck_sequential=True
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = BottleneckBlock(
                in_channels=feature_dim,
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                num_norm_groups=num_norm_groups
            )
            if bottleneck_sequential:
                bottleneck_layer = nn.Sequential(bottleneck_layer)
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))
        self.use_output_head = use_output_head

        if self.use_output_head:
            """
            0. Note that the bottleneck layers have GroupNorm and relu activations
            1. We use silu following Stable Diffusion / ControlNet
            2. We keep outputs to size 64x64 so that there aren't memory issues during guidance
            3. We normalize control to (-0.5, 0.5) and use a tanh activation since
            both depth and pose often contain extreme values (gradient saturates) but for rgb 
            images these are more rare
            """
            print(f"Using output head with {output_head_channels} channels")
            output_head = []
            output_head.extend([
                nn.Conv2d(projection_dim, 128, kernel_size=3, stride=1, padding=1),
                nn.SiLU(True),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.SiLU(True),
                nn.Conv2d(32, output_head_channels, kernel_size=1, stride=1, padding=0),
            ])
            if output_head_act:
                output_head.append(
                    nn.Tanh()
                )
            self.output_head = nn.Sequential(*output_head)
            self.output_head = self.output_head.to(device)
        else:
            print("Not using output head")
            self.output_head = nn.Identity()

    def forward(self, batch, emb=None):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            if type(bottleneck_layer) is not nn.Sequential:
                bottlenecked_feature = bottleneck_layer(feats, emb)
            else:
                bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        if self.use_output_head:
            output_feature = self.output_head(output_feature)
        return output_feature