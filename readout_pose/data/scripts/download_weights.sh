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

readout_heads=(
    # SDv1-5
    "readout_sdv15_pose_keypoint"
    # SDXL
    "readout_sdxl_pose_keypoint"
)

weights_root=weights

mkdir ${weights_root}
for readout_head in "${readout_heads[@]}"; do
    wget -O ${weights_root}/${readout_head}.pt https://huggingface.co/g-luo/readout-guidance/resolve/main/readout_pose/${readout_head}.pt?download=true
done