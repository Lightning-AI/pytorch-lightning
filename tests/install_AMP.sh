#!/usr/bin/env bash

ROOT=$PWD
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# If build with extensions fails, you can run this line to build without extensions
# pip install -v --no-cache-dir ./
cd $ROOT
rm -rf apex
