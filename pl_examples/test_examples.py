import importlib
import platform
from unittest import mock

import pytest
import torch

try:
    from nvidia.dali import ops, pipeline, plugin, types
except (ImportError, ModuleNotFoundError):
    DALI_AVAILABLE = False
else:
    DALI_AVAILABLE = True

ARGS_DEFAULT = """
--max_epochs 1 \
--batch_size 32 \
--limit_train_batches 2 \
--limit_val_batches 2 \
"""

ARGS_GPU = ARGS_DEFAULT + """
--gpus 1 \
"""

ARGS_DP_AMP = ARGS_DEFAULT + """
--gpus 2 \
--distributed_backend dp \
--precision 16 \
"""

ARGS_DDP_AMP = ARGS_DEFAULT + """
--gpus 2 \
--distributed_backend ddp \
--precision 16 \
"""


# ToDo: fix this failing example
# @pytest.mark.parametrize('import_cli', [
#     'pl_examples.basic_examples.mnist_classifier',
#     'pl_examples.basic_examples.image_classifier',
#     'pl_examples.basic_examples.autoencoder',
# ])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# @pytest.mark.parametrize('cli_args', [ARGS_DP_AMP])
# def test_examples_dp(import_cli, cli_args):
#
#     module = importlib.import_module(import_cli)
#
#     with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
#         module.cli_main()


# ToDo: fix this failing example
# @pytest.mark.parametrize('import_cli', [
#     'pl_examples.basic_examples.mnist_classifier',
#     'pl_examples.basic_examples.image_classifier',
#     'pl_examples.basic_examples.autoencoder',
# ])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# @pytest.mark.parametrize('cli_args', [ARGS_DDP_AMP])
# def test_examples_ddp(import_cli, cli_args):
#
#     module = importlib.import_module(import_cli)
#
#     with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
#         module.cli_main()


@pytest.mark.parametrize('import_cli', [
    'pl_examples.basic_examples.mnist_classifier',
    'pl_examples.basic_examples.image_classifier',
    'pl_examples.basic_examples.autoencoder',
])
@pytest.mark.parametrize('cli_args', [ARGS_DEFAULT])
def test_examples_cpu(import_cli, cli_args):

    module = importlib.import_module(import_cli)

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        module.cli_main()


@pytest.mark.skipif(not DALI_AVAILABLE, reason="Nvidia DALI required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(platform.system() != 'Linux', reason='Only applies to Linux platform.')
@pytest.mark.parametrize('cli_args', [ARGS_GPU])
def test_examples_mnist_dali(cli_args):
    from pl_examples.basic_examples.mnist_classifier_dali import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()
