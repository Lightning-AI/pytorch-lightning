from unittest import mock

import pytest
import torch


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cpu_template(cli_args):
    """Test running CLI for an example with default params."""
    from pl_examples.basic_examples.cpu_template import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_template(cli_args):
    """Test running CLI for an example with default params."""
    from pl_examples.basic_examples.gpu_template import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
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
