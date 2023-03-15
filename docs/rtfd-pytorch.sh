#!/bin/bash

export PACKAGE_NAME=pytorch
pip install -e . -r requirements/pytorch/docs.txt -f "https://download.pytorch.org/whl/cpu/torch_stable.html"
pwd
ls -lh .
cd ./docs/source-pytorch
ls -lh .
make html --jobs 2
cd ..
ls -lh ./build
mv build/html ../_readthedocs/html
