from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
])
def test_image_classifier_example(cli_args):
    from pl_examples.basic_examples.mnist import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
])
def test_mnist_example(cli_args):
    from pl_examples.basic_examples.image_classifier import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
])
def test_autoencoder_example(cli_args):
    from pl_examples.basic_examples.autoencoder import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()