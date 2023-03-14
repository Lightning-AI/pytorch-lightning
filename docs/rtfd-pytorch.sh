#!/bin/bash

export PACKAGE_NAME=pytorch
pip install -e . -r requirements/pytorch/docs.txt -f "https://download.pytorch.org/whl/cpu/torch_stable.html"
pwd
ls -lh .
ls -lh ./docs/source-pytorch
cd ./docs/source-pytorch
make html --jobs 2
ls -lh ./docs
ls -lh ./docs/build
mv docs/build/html _readthedocs/html
