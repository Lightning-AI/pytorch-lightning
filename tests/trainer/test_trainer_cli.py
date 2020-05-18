import inspect
import pickle
from argparse import ArgumentParser, Namespace
from unittest import mock

import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=Namespace(**Trainer.default_attributes()))
def test_default_args(tmpdir):
    """Tests default argument parser for Trainer"""

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
    ['--print_nan_grads', '--weights_save_path=./'],
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


# todo: add also testing for "gpus"
@pytest.mark.parametrize(['cli_args', 'expected'], [
    pytest.param('--auto_lr_find --auto_scale_batch_size power',
                 {'auto_lr_find': True, 'auto_scale_batch_size': 'power', 'early_stop_callback': False}),
    pytest.param('--auto_lr_find any_string --auto_scale_batch_size',
                 {'auto_lr_find': 'any_string', 'auto_scale_batch_size': True}),
    pytest.param('--early_stop_callback',
                 {'auto_lr_find': False, 'early_stop_callback': True, 'auto_scale_batch_size': False}),
])
def test_argparse_args_parsing(cli_args, expected):
    """Test multi type argument with bool."""
    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = Trainer.add_argparse_args(parent_parser=parser)
        args = Trainer.parse_argparser(parser)

    for k, v in expected.items():
        assert getattr(args, k) == v
    assert Trainer.from_argparse_args(args)
