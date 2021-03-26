#!/bin/bash

echo $@

full_path=$(realpath $0)
echo $full_path

dir_path=$(dirname $full_path)
echo $dir_path

python ${dir_path}/basic_examples/simple_image_classifier.py $@

python ${dir_path}/basic_examples/backbone_image_classifier.py $@

python ${dir_path}/basic_examples/autoencoder.py $@
