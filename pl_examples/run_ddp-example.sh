#!/bin/bash

ARGS_DEFAULT=" --default_root_dir %(tmpdir)s --max_epochs 1 --batch_size 32 --limit_train_batches 2 --limit_val_batches 2"
ARGS_EXTRA_DDP=" --gpus 2 --accelerator ddp"
ARGS_EXTRA_AMP=" --precision 16"

python pl_examples/basic_examples/simple_image_classifier.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP}
python pl_examples/basic_examples/simple_image_classifier.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP} ${ARGS_EXTRA_AMP}
python pl_examples/basic_examples/backbone_image_classifier.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP}
python pl_examples/basic_examples/backbone_image_classifier.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP} ${ARGS_EXTRA_AMP}
python pl_examples/basic_examples/autoencoder.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP}
python pl_examples/basic_examples/autoencoder.py ${ARGS_DEFAULT} ${ARGS_EXTRA_DDP} ${ARGS_EXTRA_AMP}
