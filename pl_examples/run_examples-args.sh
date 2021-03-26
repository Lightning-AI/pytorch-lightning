#!/bin/bash

echo $@

python pl_examples/basic_examples/simple_image_classifier.py $@

python pl_examples/basic_examples/backbone_image_classifier.py $@

python pl_examples/basic_examples/autoencoder.py $@
