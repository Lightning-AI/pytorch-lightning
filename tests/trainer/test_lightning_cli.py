# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import sys
import yaml
from argparse import Namespace
from unittest import mock

import pytest
import torch

import tests.base.develop_utils as tutils
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.cli import (
    LightningArgumentParser,
    SaveConfigCallback,
    LightningCLI
)


@mock.patch('argparse.ArgumentParser.parse_args')
def test_default_args(mock_argparse, tmpdir):
    """Tests default argument parser for Trainer"""
    mock_argparse.return_value = Namespace(**Trainer.default_attributes())

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    args = parser.parse_args([])
    args.logger = logger

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5


@pytest.mark.parametrize('cli_args', [
    ['--accumulate_grad_batches=22'],
    ['--weights_save_path=./'],
    []
])
def test_add_argparse_args_redefined(cli_args):
    """Redefines some default Trainer arguments via the cli and
    tests the Trainer initialization correctness.
    """
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_trainer_args(Trainer, None)

    args = parser.parse_args(cli_args)

    # make sure we can pickle args
    pickle.dumps(args)

    # Check few deprecated args are not in namespace:
    for depr_name in ('gradient_clip', 'nb_gpu_nodes', 'max_nb_epochs'):
        assert depr_name not in args

    trainer = Trainer.from_argparse_args(args=args)
    pickle.dumps(trainer)

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('cli_args', [
    ['--callbacks=1', '--logger'],
    ['--foo', '--bar=1']
])
def test_add_argparse_args_redefined_error(cli_args, monkeypatch):
    """Asserts error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_trainer_args(Trainer, None)

    monkeypatch.setattr(parser, 'exit', lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)


@pytest.mark.parametrize(['cli_args', 'expected'], [
    pytest.param('--auto_lr_find=True --auto_scale_batch_size=power',
                 {'auto_lr_find': True, 'auto_scale_batch_size': 'power'}),
    pytest.param('--auto_lr_find any_string --auto_scale_batch_size ON',
                 {'auto_lr_find': 'any_string', 'auto_scale_batch_size': True}),
    pytest.param('--auto_lr_find=Yes --auto_scale_batch_size=On',
                 {'auto_lr_find': True, 'auto_scale_batch_size': True}),
    pytest.param('--auto_lr_find Off --auto_scale_batch_size No',
                 {'auto_lr_find': False, 'auto_scale_batch_size': False}),
    pytest.param('--auto_lr_find TRUE --auto_scale_batch_size FALSE',
                 {'auto_lr_find': True, 'auto_scale_batch_size': False}),
    pytest.param('--tpu_cores=8',
                 {'tpu_cores': 8}),
    pytest.param('--tpu_cores=1,',
                 {'tpu_cores': '1,'}),
    pytest.param('--limit_train_batches=100',
                 {'limit_train_batches': 100}),
    pytest.param('--limit_train_batches 0.8',
                 {'limit_train_batches': 0.8}),
    pytest.param('--weights_summary=null',
                 {'weights_summary': None}),
    pytest.param(
        "",
        {
            # These parameters are marked as Optional[...] in Trainer.__init__,
            # with None as default. They should not be changed by the argparse
            # interface.
            "min_steps": None,
            "max_steps": None,
            "log_gpu_memory": None,
            "distributed_backend": None,
            "weights_save_path": None,
            "truncated_bptt_steps": None,
            "resume_from_checkpoint": None,
            "profiler": None,
        }),
])
def test_parse_args_parsing(cli_args, expected):
    """Test parsing simple types and None optionals not modified."""
    cli_args = cli_args.split(' ') if cli_args else []
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_trainer_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if 'tpu_cores' not in expected or _TPU_AVAILABLE:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(['cli_args', 'expected', 'instantiate'], [
    pytest.param(['--gpus', '[0, 2]'],
                 {'gpus': [0, 2]},
                 False),
    pytest.param(['--tpu_cores=[1,3]'],
                 {'tpu_cores': [1, 3]},
                 False),
    pytest.param(['--accumulate_grad_batches={"5":3,"10":20}'],
                 {'accumulate_grad_batches': {5: 3, 10: 20}},
                 True),
])
def test_parse_args_parsing_complex_types(cli_args, expected, instantiate):
    """Test parsing complex types."""
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_trainer_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if instantiate:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(['cli_args', 'expected_gpu'], [
    pytest.param('--gpus 1', [0]),
    pytest.param('--gpus 0,', [0]),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_parse_args_parsing_gpus(cli_args, expected_gpu):
    """Test parsing of gpus and instantiation of Trainer."""
    cli_args = cli_args.split(' ') if cli_args else []
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_trainer_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)
    assert trainer.data_parallel_device_ids == expected_gpu


@pytest.mark.skipif(sys.version_info < (3, 7),
                    reason="signature inspection while mocking is not working in Python < 3.7 despite autospec")
@pytest.mark.parametrize(['cli_args', 'extra_args'], [
    pytest.param({}, {}),
    pytest.param({'logger': False}, {}),
    pytest.param({'logger': False}, {'logger': True}),
    pytest.param({'logger': False}, {'checkpoint_callback': True}),
])
def test_init_from_argparse_args(cli_args, extra_args):
    unknown_args = dict(unknown_arg=0)

    # unkown args in the argparser/namespace should be ignored
    with mock.patch('pytorch_lightning.Trainer.__init__', autospec=True, return_value=None) as init:
        trainer = Trainer.from_argparse_args(Namespace(**cli_args, **unknown_args), **extra_args)
        expected = dict(cli_args)
        expected.update(extra_args)  # extra args should override any cli arg
        init.assert_called_with(trainer, **expected)

    # passing in unknown manual args should throw an error
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'unknown_arg'"):
        Trainer.from_argparse_args(Namespace(**cli_args), **extra_args, **unknown_args)


@pytest.mark.parametrize(['cli_args', 'expected_model', 'expected_trainer'], [
    pytest.param(['--model.model_param=7', '--trainer.limit_train_batches=100'],
                 {'model_param': 7},
                 {'limit_train_batches': 100}),
])
def test_lightning_cli(cli_args, expected_model, expected_trainer, monkeypatch):
    """Test that LightningCLI correctly instantiates model, trainer and calls fit."""

    def fit(trainer, model):
        for k, v in model.expected_model.items():
            assert getattr(model, k) == v
        for k, v in model.expected_trainer.items():
            assert getattr(trainer, k) == v
        save_callback = [x for x in trainer.callbacks if isinstance(x, SaveConfigCallback)]
        assert len(save_callback) == 1
        save_callback[0].on_train_start(trainer, model)

    def on_train_start(callback, trainer, model):
        config_dump = callback.parser.dump(callback.config, skip_none=False)
        for k, v in model.expected_model.items():
            assert f'  {k}: {v}' in config_dump
        for k, v in model.expected_trainer.items():
            assert f'  {k}: {v}' in config_dump
        trainer.ran_asserts = True

    monkeypatch.setattr(Trainer, 'fit', fit)
    monkeypatch.setattr(SaveConfigCallback, 'on_train_start', on_train_start)

    class TestModel(LightningModule):
        def __init__(self, model_param: int):
            super().__init__()
            self.model_param = model_param

    TestModel.expected_model = expected_model
    TestModel.expected_trainer = expected_trainer

    with mock.patch('sys.argv', ['any.py'] + cli_args):
        cli = LightningCLI(TestModel, trainer_class=Trainer, save_config_callback=SaveConfigCallback)
        assert hasattr(cli.trainer, 'ran_asserts') and cli.trainer.ran_asserts


class TestLightningCLI(LightningCLI):
    def before_fit(self):
        for key in ['validation_step',
                    'validation_step_end',
                    'validation_epoch_end',
                    'test_step',
                    'test_step_end',
                    'test_epoch_end']:
            setattr(self.model, key, None)


def test_lightning_cli_mnist_args(tmpdir):

    cli_args = [
        '--data.data_dir=' + str(tmpdir),
        '--trainer.default_root_dir=' + str(tmpdir),
        '--trainer.max_epochs=1',
        '--trainer.weights_summary=null',
    ]

    with mock.patch('sys.argv', ['trial.py'] + cli_args):
        cli = TestLightningCLI(EvalModelTemplate, TrialMNISTDataModule)
        assert cli.fit_result == 1
        config_path = os.path.join(str(tmpdir), 'lightning_logs', 'version_0', 'config.yaml')
        assert os.path.isfile(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f.read())
        assert config['model'] == cli.config['model']
        assert config['data'] == cli.config['data']
        assert config['trainer'] == cli.config['trainer']


def test_lightning_cli_mnist_config_and_subclass_mode(tmpdir):

    config = {
        'model': {
            'class_path': 'tests.base.EvalModelTemplate',
        },
        'data': {
            'class_path': 'tests.base.datamodules.TrialMNISTDataModule',
            'init_args': {
                'data_dir': str(tmpdir),
            },
        },
        'trainer': {
            'default_root_dir': str(tmpdir),
            'max_epochs': 1,
            'weights_summary': None,
        },
    }
    config_path = os.path.join(str(tmpdir), 'config.yaml')
    with open(config_path, 'w') as f:
        f.write(yaml.dump(config))

    with mock.patch('sys.argv', ['trial.py', '--config', config_path]):
        cli = TestLightningCLI(
            EvalModelTemplate,
            TrialMNISTDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True
        )
        assert cli.fit_result == 1
        config_path = os.path.join(str(tmpdir), 'lightning_logs', 'version_0', 'config.yaml')
        assert os.path.isfile(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f.read())
        assert config['model'] == cli.config['model']
        assert config['data'] == cli.config['data']
        assert config['trainer'] == cli.config['trainer']
