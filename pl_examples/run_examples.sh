#!/bin/bash
set -ex

dir_path=$(dirname "${BASH_SOURCE[0]}")
args="
  --data.batch_size=32
  --trainer.max_epochs=1
  --trainer.limit_train_batches=2
  --trainer.limit_val_batches=2
  --trainer.limit_test_batches=2
  --trainer.limit_predict_batches=2
"

python "${dir_path}/basic_examples/simple_image_classifier.py" ${args} "$@"
python "${dir_path}/basic_examples/backbone_image_classifier.py" ${args} "$@"
python "${dir_path}/basic_examples/autoencoder.py" ${args} "$@"
