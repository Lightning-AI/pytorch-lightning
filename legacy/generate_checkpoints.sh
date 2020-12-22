#!/bin/bash

# todo: take this from argument
VERSIONS=("1.0.0" "1.0.1" "1.0.2")

LEGACY_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo $LEGACY_PATH
# install some PT version here so it does not need to reinstalled for each env
pip install virtualenv "torch==1.5" --quiet --no-cache-dir

ENV_PATH="$LEGACY_PATH/vEnv"

for ver in "${VERSIONS[@]}"
do
	echo "processing version: $ver"
	# mkdir "$LEGACY_PATH/$ver"

  # create local env
  echo $ENV_PATH
  virtualenv $ENV_PATH --system-site-packages
  # activate and install PL version
  source "$ENV_PATH/bin/activate"
  pip install "pytorch_lightning==$ver" --quiet -U --no-cache-dir

  python --version
  pip --version
  pip list | grep torch

  python "$LEGACY_PATH/zero_training.py"
  cp "$LEGACY_PATH/zero_training.py" ${LEGACY_PATH}/checkpoints/${ver}

  mv ${LEGACY_PATH}/checkpoints/${ver}/lightning_logs/version_0/checkpoints/*.ckpt ${LEGACY_PATH}/checkpoints/${ver}/
  rm -rf ${LEGACY_PATH}/checkpoints/${ver}/lightning_logs

  deactivate
  # clear env
  rm -rf $ENV_PATH

done
