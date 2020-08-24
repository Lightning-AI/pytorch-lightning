import inspect
import pytest
from argparse import ArgumentParser
import pickle
from unittest import mock

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.hypertuner import HyperTuner
from tests.base import EvalModelTemplate


@pytest.mark.parametrize(['first_method', 'second_method'], [
    pytest.param('lr_find', 'scale_batch_size'),
    pytest.param('n_worker_search', 'scale_batch_size')
])
def test_call_order(tmpdir, first_method, second_method):
    """ Check that an warning occurs if the methods are called in a different
        order than expected """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    tuner = HyperTuner(trainer, model)

    # Wrong call order, should give warning
    with pytest.warns(UserWarning):
        _ = getattr(tuner, first_method)()
        _ = getattr(tuner, second_method)()


def test_get_init_arguments_and_types():
    """Asserts a correctness of the `get_init_arguments_and_types` Trainer classmethod."""
    args = HyperTuner.get_init_arguments_and_types()
    parameters = inspect.signature(HyperTuner).parameters
    assert len(parameters) - 2 == len(args)  # subtract trainer and model
    for arg in args:
        assert parameters[arg[0]].default == arg[2]

    kwargs = {arg[0]: arg[2] for arg in args}
    tuner = HyperTuner(Trainer(), EvalModelTemplate(), **kwargs)
    assert isinstance(tuner, HyperTuner)


@pytest.mark.parametrize(['cli_args', 'expected'], [
    pytest.param('--auto_lr_find --auto_scale_batch_size power',
                 {'auto_lr_find': True, 'auto_scale_batch_size': 'power', 'auto_n_worker_search': False}),
    pytest.param('--auto_lr_find any_string --auto_scale_batch_size',
                 {'auto_lr_find': 'any_string', 'auto_scale_batch_size': True, 'auto_n_worker_search': False}),
])
def test_argparse_args_parsing(cli_args, expected):
    """Test multi type argument with bool."""
    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        parser = ArgumentParser(add_help=False)
        parser = HyperTuner.add_argparse_args(parent_parser=parser)
        arguments = HyperTuner.parse_argparser(parser)

    for k, v in expected.items():
        assert getattr(arguments, k) == v
    assert HyperTuner.from_argparse_args(arguments,
                                         trainer=Trainer(),
                                         model=EvalModelTemplate())
