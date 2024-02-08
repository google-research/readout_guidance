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

data_root=data/raw

mkdir ${data_root}

# Download PascalVOC 2012 images
wget -P ${data_root} http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf ${data_root}/VOCtrainval_11-May-2012.tar -C ${data_root}
mv ${data_root}/VOCdevkit ${data_root}/PascalVOC
rm -rf ${data_root}/VOCtrainval_11-May-2012.tar

# Download DAVIS 2017 videos
wget -P ${data_root} https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip ${data_root}/DAVIS-2017-trainval-480p.zip -d ${data_root}
rm -rf ${data_root}/DAVIS-2017-trainval-480p.zip