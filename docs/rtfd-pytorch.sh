#!/bin/bash

pwd
ls -lh .
export PACKAGE_NAME=pytorch
pip install -e . -r requirements/pytorch/docs.txt -f "https://download.pytorch.org/whl/cpu/torch_stable.html"
pip list

cd ./docs/source-pytorch
ls -lh .
make html --jobs $(nproc)

cd ..
ls -lh ./build
mv build/html ../_readthedocs/html
