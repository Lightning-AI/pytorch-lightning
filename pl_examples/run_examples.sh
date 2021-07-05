#!/bin/bash

ARGS="--trainer.max_epochs=1 --data.batch_size=32 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2"
ARGS_AMP="--trainer.precision=16"

python pl_examples/basic_examples/simple_image_classifier.py $@ ${ARGS}
python pl_examples/basic_examples/simple_image_classifier.py $@ ${ARGS} ${ARGS_AMP}

python pl_examples/basic_examples/backbone_image_classifier.py $@ ${ARGS}
python pl_examples/basic_examples/backbone_image_classifier.py $@ ${ARGS} ${ARGS_AMP}

python pl_examples/basic_examples/autoencoder.py $@ ${ARGS}
python pl_examples/basic_examples/autoencoder.py $@ ${ARGS} ${ARGS_AMP}
