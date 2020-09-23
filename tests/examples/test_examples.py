from unittest import mock
import torch
import pytest

dp_16_args = """
--max_epochs 1 
--batch_size 32 
--limit_train_batches 2 
--limit_val_batches 2 
--gpus 2 
--distributed_backend dp 
--precision 16
"""

cpu_args = """
--max_epochs 1 
--batch_size 32 
--limit_train_batches 2 
--limit_val_batches 2 
"""

ddp_args = """
--max_epochs 1 
--batch_size 32 
--limit_train_batches 2 
--limit_val_batches 2 
--gpus 2 
--precision 16
"""

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize('cli_args', [dp_16_args, cpu_args, ddp_args])
def test_image_classifier_example(cli_args):
    from pl_examples.basic_examples.mnist import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize('cli_args', [dp_16_args, cpu_args, ddp_args])
def test_mnist_example(cli_args):
    from pl_examples.basic_examples.image_classifier import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize('cli_args', [dp_16_args, cpu_args, ddp_args])
def test_autoencoder_example(cli_args):
    from pl_examples.basic_examples.autoencoder import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()
