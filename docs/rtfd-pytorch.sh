#!/bin/bash

pwd
python --version
pip --version
ls -lh .
export PACKAGE_NAME=pytorch
sudo pip install -q -r requirements/pytorch/docs.txt
pip install -e . -q -f "https://download.pytorch.org/whl/cpu/torch_stable.html"
pip list

cd ./docs/source-pytorch
ls -lh .
make html --jobs $(nproc)

cd ..
ls -lh ./build
mv build/html ../_readthedocs/html
