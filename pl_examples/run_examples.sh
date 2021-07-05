#!/bin/bash
set -ex

ARGS="--trainer.max_epochs=1 --data.batch_size=32 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2"

python pl_examples/basic_examples/simple_image_classifier.py $@ ${ARGS}
python pl_examples/basic_examples/backbone_image_classifier.py $@ ${ARGS}
python pl_examples/basic_examples/autoencoder.py $@ ${ARGS}
