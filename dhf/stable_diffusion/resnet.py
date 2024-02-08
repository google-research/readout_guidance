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
# This code is adapted from resnet.py in Diffusion Hyperfeatures
# and implements resnet feature caching.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/stable_diffusion/resnet.py
# =================================================================

import torch

def init_resnet_func(
    unet,
    save_mode="",
    reset=True,
    idxs=None
):
    def new_forward(self, input_tensor, temb, scale=None):
        # https://github.com/huggingface/diffusers/blob/20e92586c1fda968ea3343ba0f44f2b21f3c09d2/src/diffusers/models/resnet.py#L460
        if save_mode == "input":
            self.feats = input_tensor

        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        
        if save_mode == "hidden":
            self.feats = hidden_states

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        if save_mode == "output":
            self.feats = output_tensor

        self.hws = (output_tensor.shape[2], output_tensor.shape[3])
        return output_tensor

    layers = collect_layers(unet, idxs)
    for module in layers:
        module.forward = new_forward.__get__(module, type(module))
        if reset:
            module.feats = None
            module.hws = None

def collect_layers(unet, idxs=None):
    layers = []
    layer_idx = 0
    for up_block in unet.up_blocks:
        for module in up_block.resnets:
            if idxs is None or layer_idx in idxs:
                layers.append(module)
            layer_idx += 1
    return layers

def collect_feats(unet, idxs=None):
    return [module.feats for module in collect_layers(unet, idxs)]

def collect_channels(unet, idxs=None):
    return [module.time_emb_proj.out_features for module in collect_layers(unet, idxs)]

def collect_hws(unet, idxs=None):
    return [module.hws for module in collect_layers(unet, idxs)]

def set_timestep(unet, timestep=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        module.timestep = timestep