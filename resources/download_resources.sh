#!/bin/bash
dir=$(dirname "$0")
echo Downloading resources to directory ... $dir

# object detection model
modelfile=ssd_mobilenet_v1_coco_2017_11_17.tar.gz
modelpath=http://download.tensorflow.org/models/object_detection/$modelfile
wget -P $dir $modelpath
tar -xvzf $dir/$modelfile -C $dir

# fid script
fid=https://github.com/bioinf-jku/TTUR/blob/465c8fed9358edd3c4119b751608d8af45a835d5/fid.py
wget -P $dir $fid
chmod u+x $dir/fid.py

# stylegan model
git clone https://github.com/NVlabs/stylegan.git
mv ./stylegan $dir

# lpips
git clone https://github.com/alexlee-gk/lpips-tensorflow.git
mv ./lpips-tensorflow $dir
