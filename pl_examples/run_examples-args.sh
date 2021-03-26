#!/bin/bash

echo $@

ARGS_DEFAULT=" --default_root_dir %(tmpdir)s --max_epochs 1 --batch_size 32 --limit_train_batches 2 --limit_val_batches 2"

python pl_examples/basic_examples/simple_image_classifier.py ${ARGS_DEFAULT} $@
python pl_examples/basic_examples/backbone_image_classifier.py ${ARGS_DEFAULT} $@
python pl_examples/basic_examples/autoencoder.py ${ARGS_DEFAULT} $@
