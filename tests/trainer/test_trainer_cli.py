import inspect
from argparse import ArgumentParser, Namespace
from unittest import mock
import pickle

import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=Namespace(**Trainer.default_attributes()))
def test_default_args(tmpdir):
    """Tests default argument parser for Trainer"""
    tutils.reset_seed()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    parser = ArgumentParser(add_help=False)
    args = parser.parse_args()
    args.logger = logger

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5


@pytest.mark.parametrize('cli_args', [
    ['--accumulate_grad_batches=22'],
    ['--print_nan_grads=1', '--weights_save_path=./'],
    []
])
def test_add_argparse_args_redefined(cli_args):
    """Redefines some default Trainer arguments via the cli and
    tests the Trainer initialization correctness.
    """
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args(cli_args)

    # make sure we can pickle args
    pickle.dumps(args)

    # Check few deprecated args are not in namespace:
    for depr_name in ('gradient_clip', 'nb_gpu_nodes', 'max_nb_epochs'):
        assert depr_name not in args

    trainer = Trainer.from_argparse_args(args=args)
    pickle.dumps(trainer)

    assert isinstance(trainer, Trainer)


def test_get_init_arguments_and_types():
    """Asserts a correctness of the `get_init_arguments_and_types` Trainer classmethod."""
    args = Trainer.get_init_arguments_and_types()
    parameters = inspect.signature(Trainer).parameters
    assert len(parameters) == len(args)
    for arg in args:
        assert parameters[arg[0]].default == arg[2]

    kwargs = {arg[0]: arg[2] for arg in args}
    trainer = Trainer(**kwargs)
    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('cli_args', [
    ['--callbacks=1', '--logger'],
    ['--foo', '--bar=1']
])
def test_add_argparse_args_redefined_error(cli_args, monkeypatch):
    """Asserts thar an error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parent_parser=parser)

    monkeypatch.setattr(parser, 'exit', lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)
