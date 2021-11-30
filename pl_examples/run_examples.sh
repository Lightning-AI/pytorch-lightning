#!/bin/bash
set -ex

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")
args="
  --data.batch_size=32
  --trainer.max_epochs=1
  --trainer.limit_train_batches=2
  --trainer.limit_val_batches=2
  --trainer.limit_test_batches=2
  --trainer.limit_predict_batches=2
  --optimizer=Adam
"

python "${dir_path}/basic_examples/backbone_image_classifier.py" ${args} "$@"
python "${dir_path}/basic_examples/autoencoder.py" ${args} "$@"


args="--dry-run"
python "${dir_path}/basic_examples/mnist_examples/image_classifier_1_pytorch.py" ${args}
python "${dir_path}/basic_examples/mnist_examples/image_classifier_2_lite.py" ${args}
python "${dir_path}/basic_examples/mnist_examples/image_classifier_3_lite_to_lightning_module.py" ${args}
python "${dir_path}/loop_examples/mnist_lite.py" ${args}


args="
  --trainer.max_epochs=1
  --trainer.limit_train_batches=2
  --trainer.limit_val_batches=2
  --trainer.limit_test_batches=2
  --trainer.limit_predict_batches=2
  --optimizer=Adam
"

python "${dir_path}/basic_examples/mnist_examples/image_classifier_4_lightning_module.py" ${args} "$@"
python "${dir_path}/basic_examples/mnist_examples/image_classifier_5_lightning_datamodule.py" ${args} "$@"

python "${dir_path}/fault_tolerant/automatic.py"
