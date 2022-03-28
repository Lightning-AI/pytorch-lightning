#!/bin/bash
# Sample call:
#  bash generate_checkpoints.sh 1.0.2 1.0.3 1.0.4
set -e

LEGACY_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENV_PATH="$LEGACY_PATH/vEnv"

# iterate over all arguments assuming that each argument is version
for pl_ver in "$@"
do
	echo processing version: $pl_ver

  # Don't install/update anything before activating venv
  # to avoid breaking any existing environment.
  python -m venv $ENV_PATH
  source $ENV_PATH/bin/activate

  # there are problem to load ckpt in older versions since they are saved the newer versions
  python -m pip install "pytorch_lightning==$pl_ver" torch "torchmetrics" "scikit-learn" --progress-bar off

  python --version
  python -m pip --version
  python -m pip list | grep -e torch -e learn

  python "$LEGACY_PATH/simple_classif_training.py" > /dev/null 2> &1
  cp "$LEGACY_PATH/simple_classif_training.py" ${LEGACY_PATH}/checkpoints/$pl_ver

  mv ${LEGACY_PATH}/checkpoints/$pl_ver/lightning_logs/version_0/checkpoints/*.ckpt $LEGACY_PATH/checkpoints/$pl_ver/
  rm -rf ${LEGACY_PATH}/checkpoints/$pl_ver/lightning_logs

  deactivate
  rm -rf $ENV_PATH
done
