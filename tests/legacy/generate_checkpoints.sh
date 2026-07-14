#!/bin/bash
# Usage:
# 1. Generate checkpoints with one or more specified PL versions:
#    bash generate_checkpoints.sh 1.0.2 1.0.3 1.0.4
# 2. Generate checkpoints with the PL version installed in your environment:
#    bash generate_checkpoints.sh
set -e

LEGACY_FOLDER=$(cd $(dirname $0); pwd -P)
printf "LEGACY_FOLDER: $LEGACY_FOLDER\n"
TESTS_FOLDER=$(dirname $LEGACY_FOLDER)
ENV_PATH=$LEGACY_FOLDER/.venv
printf "ENV_PATH: $ENV_PATH\n"
export PYTHONPATH=$TESTS_FOLDER  # for `import tests_pytorch`
printf "PYTHONPATH: $PYTHONPATH\n"
rm -rf $ENV_PATH

function create_and_save_checkpoint {
  uv --version
  uv pip list

  python $LEGACY_FOLDER/simple_classif_training.py $pl_ver

  cp $LEGACY_FOLDER/simple_classif_training.py $LEGACY_FOLDER/checkpoints/$pl_ver
  mv $LEGACY_FOLDER/checkpoints/$pl_ver/lightning_logs/version_0/checkpoints/*.ckpt $LEGACY_FOLDER/checkpoints/$pl_ver/
  rm -rf $LEGACY_FOLDER/checkpoints/$pl_ver/lightning_logs
}

# iterate over all arguments assuming that each argument is version
for pl_ver in "$@"
do
  printf "\n\n processing version: $pl_ver\n"

  # Don't install/update anything before activating venv to avoid breaking any existing environment.
  uv venv $ENV_PATH
  source $ENV_PATH/bin/activate

  uv pip install "pytorch_lightning==$pl_ver" \
    -r "$(dirname $TESTS_FOLDER)/requirements/pytorch/test.txt" \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

  rm -rf $LEGACY_FOLDER/checkpoints/$pl_ver
  create_and_save_checkpoint

  deactivate
  rm -rf $ENV_PATH
done

# use the PL installed in the environment if no PL version is specified
if [[ -z "$@" ]]; then
  printf "\n\n processing local version\n"

  uv pip install \
    -r "$(dirname $TESTS_FOLDER)/requirements/pytorch/test.txt" \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
  pl_ver="local"
  create_and_save_checkpoint
fi
