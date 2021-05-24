#!/bin/bash
# on master branch 01109cdf0c44a150c262b65e70a7e1e64003cf93 commit

ARGS_EXTRA_DDP=" --trainer.gpus 2 --trainer.accelerator ddp"
ARGS_EXTRA_AMP=" --trainer.precision 16"

# conda created with the following command in the git root directory
#   conda env create --file environment.yml --prefix $PWD/env
#   conda activate $PWD/env
#   pip install pytorch-lightning==1.3.2
conda activate ../env
python basic_examples/autoencoder.py --config basic_examples/autoencoder.yml ${ARGS_EXTRA_DDP} ${ARGS_EXTRA_AMP} $@
