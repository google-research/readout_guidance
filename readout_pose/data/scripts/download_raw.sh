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

data_root=data/raw/MSCOCO

# Download MSCOCO annotations
mkdir ${data_root}
wget -P ${data_root} http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip ${data_root}/annotations_trainval2017.zip -d ${data_root}
rm -rf ${data_root}/annotations_trainval2017.zip

data_root=data/raw/MSCOCO/images

# Download MSCOCO train2017 images
mkdir ${data_root}
wget -P ${data_root} http://images.cocodataset.org/zips/train2017.zip
unzip ${data_root}/train2017.zip -d ${data_root}
rm -rf ${data_root}/train2017.zip

# Download MSCOCO val2017 images
mkdir ${data_root}
wget -P ${data_root} http://images.cocodataset.org/zips/val2017.zip
unzip ${data_root}/val2017.zip -d ${data_root}
rm -rf ${data_root}/val2017.zip