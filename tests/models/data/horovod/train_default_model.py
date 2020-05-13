"""
This script is meant to be executed from `../../test_horovod.py`.

Because Horovod uses a parallel programming model similar to MPI, unit tests for collective
ops like allreduce need to be run in parallel. The most common approach for running parallel
Horovod workers is to launch multiple replicas of the training script via the `horovodrun`
command-line tool:

.. code-block:: bash

    horovodrun -np 2 python train_default_model.py ...

Individual test parameters are configured by the serialized `--trainer-options` JSON object.

An non-zero exit code from this script on any rank will indicate failure, while a zero exit code
across all ranks indicates success.
"""

import argparse
import json
import os
import sys

import horovod.torch as hvd

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))

from pytorch_lightning import Trainer  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from tests.base import EvalModelTemplate  # noqa: E402
from tests.base.utils import set_random_master_port, run_model_test  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument('--trainer-options', required=True)
parser.add_argument('--on-gpu', action='store_true', default=False)


def run_test_from_config(trainer_options):
    """Trains the default model with the given config."""
    set_random_master_port()

    ckpt_path = trainer_options['default_root_dir']
    trainer_options.update(checkpoint_callback=ModelCheckpoint(ckpt_path))

    model = EvalModelTemplate()
    run_model_test(trainer_options, model, on_gpu=args.on_gpu, version=0, with_hpc=False)

    # Horovod should be initialized following training. If not, this will raise an exception.
    assert hvd.size() == 2

    if args.on_gpu:
        trainer = Trainer(gpus=1, distributed_backend='horovod', max_epochs=1)
        # Test the root_gpu property
        assert trainer.root_gpu == hvd.local_rank()


if __name__ == "__main__":
    args = parser.parse_args()
    run_test_from_config(json.loads(args.trainer_options))
