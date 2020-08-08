import os
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cpu_template(cli_args):
    """Test running CLI for an example with default params."""
    from pl_examples.basic_examples.cpu_template import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --gpus 1'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_template(cli_args):
    """Test running CLI for an example with default params."""
    from pl_examples.basic_examples.gpu_template import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --gpus 1',
    '--max_epochs 1 --gpus 1 --evaluate',
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_imagenet(tmpdir, cli_args):
    """Test running CLI for the ImageNet example with default params."""

    from pl_examples.domain_templates.imagenet import run_cli

    # https://github.com/pytorch/vision/blob/master/test/fakedata_generation.py#L105
    def _make_image(file_path):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(file_path)

    for split in ['train', 'val']:
        for class_id in ['a', 'b']:
            os.makedirs(os.path.join(tmpdir, split, class_id))
            # Generate 5 black images
            for image_id in range(5):
                _make_image(os.path.join(tmpdir, split, class_id, str(image_id) + '.JPEG'))

    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--data-path', str(tmpdir)]
    cli_args += ['--default_root_dir', str(tmpdir)]

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


# @pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --num_nodes 1 --gpus 2'])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# def test_multi_node_ddp(cli_args):
#     """Test running CLI for an example with default params."""
#     from pl_examples.basic_examples.multi_node_ddp_demo import run_cli
#
#     cli_args = cli_args.split(' ') if cli_args else []
#     with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
#         run_cli()


# @pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --num_nodes 1 --gpus 2'])
# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
# def test_multi_node_ddp2(cli_args):
#     """Test running CLI for an example with default params."""
#     from pl_examples.basic_examples.multi_node_ddp2_demo import run_cli
#
#     cli_args = cli_args.split(' ') if cli_args else []
#     with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
#         run_cli()
