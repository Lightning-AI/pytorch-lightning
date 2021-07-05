#!/bin/bash
set -ex

full_path=$(realpath $0)
dir_path=$(dirname $full_path)
args="--trainer.max_epochs=1 --data.batch_size=32 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2"

python ${dir_path}/basic_examples/simple_image_classifier.py $@ ${args}
python ${dir_path}/basic_examples/backbone_image_classifier.py $@ ${args}
python ${dir_path}/basic_examples/autoencoder.py $@ ${args}
