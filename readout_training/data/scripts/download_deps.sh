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

deps_root=data/deps

mkdir ${deps_root}

# Clone ControlNet and create `control` conda env
git clone https://github.com/lllyasviel/ControlNet ${deps_root}/ControlNet
conda env create -f ${deps_root}/ControlNet/environment.yaml

# CoTracker is already included via torch
# and is compatable with the `readout` conda env

# SDEdit is already included via diffusers
# and is compatable with the `readout` conda env