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

import inspect
import json
import os
import pickle
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import pytest
import yaml

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from tests.helpers import BoringDataModule, BoringModel


@mock.patch('argparse.ArgumentParser.parse_args')
def test_default_args(mock_argparse, tmpdir):
    """Tests default argument parser for Trainer"""
    mock_argparse.return_value = Namespace(**Trainer.default_attributes())

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    args = parser.parse_args([])

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5


@pytest.mark.parametrize('cli_args', [['--accumulate_grad_batches=22'], ['--weights_save_path=./'], []])
def test_add_argparse_args_redefined(cli_args):
    """Redefines some default Trainer arguments via the cli and
    tests the Trainer initialization correctness.
    """
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)

    args = parser.parse_args(cli_args)

    # make sure we can pickle args
    pickle.dumps(args)

    # Check few deprecated args are not in namespace:
    for depr_name in ('gradient_clip', 'nb_gpu_nodes', 'max_nb_epochs'):
        assert depr_name not in args

    trainer = Trainer.from_argparse_args(args=args)
    pickle.dumps(trainer)

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('cli_args', [['--callbacks=1', '--logger'], ['--foo', '--bar=1']])
def test_add_argparse_args_redefined_error(cli_args, monkeypatch):
    """Asserts error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)

    monkeypatch.setattr(parser, 'exit', lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)


@pytest.mark.parametrize(
    ['cli_args', 'expected'],
    [
        ('--auto_lr_find=True --auto_scale_batch_size=power', dict(auto_lr_find=True, auto_scale_batch_size='power')),
        (
            '--auto_lr_find any_string --auto_scale_batch_size ON',
            dict(auto_lr_find='any_string', auto_scale_batch_size=True),
        ),
        ('--auto_lr_find=Yes --auto_scale_batch_size=On', dict(auto_lr_find=True, auto_scale_batch_size=True)),
        ('--auto_lr_find Off --auto_scale_batch_size No', dict(auto_lr_find=False, auto_scale_batch_size=False)),
        ('--auto_lr_find TRUE --auto_scale_batch_size FALSE', dict(auto_lr_find=True, auto_scale_batch_size=False)),
        ('--tpu_cores=8', dict(tpu_cores=8)),
        ('--tpu_cores=1,', dict(tpu_cores='1,')),
        ('--limit_train_batches=100', dict(limit_train_batches=100)),
        ('--limit_train_batches 0.8', dict(limit_train_batches=0.8)),
        ('--weights_summary=null', dict(weights_summary=None)),
        (
            "",
            dict(
                # These parameters are marked as Optional[...] in Trainer.__init__,
                # with None as default. They should not be changed by the argparse
                # interface.
                min_steps=None,
                max_steps=None,
                log_gpu_memory=None,
                distributed_backend=None,
                weights_save_path=None,
                truncated_bptt_steps=None,
                resume_from_checkpoint=None,
                profiler=None
            ),
        ),
    ],
)
def test_parse_args_parsing(cli_args, expected):
    """Test parsing simple types and None optionals not modified."""
    cli_args = cli_args.split(' ') if cli_args else []
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if 'tpu_cores' not in expected or _TPU_AVAILABLE:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(
    ['cli_args', 'expected', 'instantiate'],
    [
        (['--gpus', '[0, 2]'], dict(gpus=[0, 2]), False),
        (['--tpu_cores=[1,3]'], dict(tpu_cores=[1, 3]), False),
        (['--accumulate_grad_batches={"5":3,"10":20}'], dict(accumulate_grad_batches={
            5: 3,
            10: 20
        }), True),
    ],
)
def test_parse_args_parsing_complex_types(cli_args, expected, instantiate):
    """Test parsing complex types."""
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if instantiate:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(
    ['cli_args', 'expected_gpu'],
    [
        ('--gpus 1', [0]),
        ('--gpus 0,', [0]),
        ('--gpus 0,1', [0, 1]),
    ],
)
def test_parse_args_parsing_gpus(monkeypatch, cli_args, expected_gpu):
    """Test parsing of gpus and instantiation of Trainer."""
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    cli_args = cli_args.split(' ') if cli_args else []
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)
    assert trainer.data_parallel_device_ids == expected_gpu


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason="signature inspection while mocking is not working in Python < 3.7 despite autospec",
)
@pytest.mark.parametrize(
    ['cli_args', 'extra_args'],
    [
        ({}, {}),
        (dict(logger=False), {}),
        (dict(logger=False), dict(logger=True)),
        (dict(logger=False), dict(checkpoint_callback=True)),
    ],
)
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


@pytest.mark.parametrize(['cli_args', 'expected_model', 'expected_trainer'], [(
    ['--model.model_param=7', '--trainer.limit_train_batches=100'],
    dict(model_param=7),
    dict(limit_train_batches=100),
)])
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


def test_lightning_cli_args_callbacks(tmpdir):

    callbacks = [
        dict(
            class_path='pytorch_lightning.callbacks.LearningRateMonitor',
            init_args=dict(logging_interval='epoch', log_momentum=True)
        ),
        dict(class_path='pytorch_lightning.callbacks.ModelCheckpoint', init_args=dict(monitor='NAME')),
    ]

    class TestModel(BoringModel):

        def on_fit_start(self):
            callback = [c for c in self.trainer.callbacks if isinstance(c, LearningRateMonitor)]
            assert len(callback) == 1
            assert callback[0].logging_interval == 'epoch'
            assert callback[0].log_momentum is True
            callback = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            assert len(callback) == 1
            assert callback[0].monitor == 'NAME'
            self.trainer.ran_asserts = True

    with mock.patch('sys.argv', ['any.py', f'--trainer.callbacks={json.dumps(callbacks)}']):
        cli = LightningCLI(TestModel, trainer_defaults=dict(default_root_dir=str(tmpdir), fast_dev_run=True))

    assert cli.trainer.ran_asserts


def test_lightning_cli_args_cluster_environments(tmpdir):
    plugins = [dict(class_path='pytorch_lightning.plugins.environments.SLURMEnvironment')]

    class TestModel(BoringModel):

        def on_fit_start(self):
            # Ensure SLURMEnvironment is set, instead of default LightningEnvironment
            assert isinstance(self.trainer.accelerator_connector._cluster_environment, SLURMEnvironment)
            self.trainer.ran_asserts = True

    with mock.patch('sys.argv', ['any.py', f'--trainer.plugins={json.dumps(plugins)}']):
        cli = LightningCLI(TestModel, trainer_defaults=dict(default_root_dir=str(tmpdir), fast_dev_run=True))

    assert cli.trainer.ran_asserts


def test_lightning_cli_args(tmpdir):

    cli_args = [
        f'--data.data_dir={tmpdir}',
        f'--trainer.default_root_dir={tmpdir}',
        '--trainer.max_epochs=1',
        '--trainer.weights_summary=null',
        '--seed_everything=1234',
    ]

    with mock.patch('sys.argv', ['any.py'] + cli_args):
        cli = LightningCLI(BoringModel, BoringDataModule, trainer_defaults={'callbacks': [LearningRateMonitor()]})

    assert cli.config['seed_everything'] == 1234
    config_path = tmpdir / 'lightning_logs' / 'version_0' / 'config.yaml'
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f.read())
    assert 'model' not in config and 'model' not in cli.config  # no arguments to include
    assert config['data'] == cli.config['data']
    assert config['trainer'] == cli.config['trainer']


def test_lightning_cli_config_and_subclass_mode(tmpdir):

    config = dict(
        model=dict(class_path='tests.helpers.BoringModel'),
        data=dict(class_path='tests.helpers.BoringDataModule', init_args=dict(data_dir=str(tmpdir))),
        trainer=dict(default_root_dir=str(tmpdir), max_epochs=1, weights_summary=None)
    )
    config_path = tmpdir / 'config.yaml'
    with open(config_path, 'w') as f:
        f.write(yaml.dump(config))

    with mock.patch('sys.argv', ['any.py', '--config', str(config_path)]):
        cli = LightningCLI(
            BoringModel,
            BoringDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            trainer_defaults={'callbacks': LearningRateMonitor()}
        )

    config_path = tmpdir / 'lightning_logs' / 'version_0' / 'config.yaml'
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f.read())
    assert config['model'] == cli.config['model']
    assert config['data'] == cli.config['data']
    assert config['trainer'] == cli.config['trainer']


def any_model_any_data_cli():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


def test_lightning_cli_help():

    cli_args = ['any.py', '--help']
    out = StringIO()
    with mock.patch('sys.argv', cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    assert '--print_config' in out.getvalue()
    assert '--config' in out.getvalue()
    assert '--seed_everything' in out.getvalue()
    assert '--model.help' in out.getvalue()
    assert '--data.help' in out.getvalue()

    skip_params = {'self'}
    for param in inspect.signature(Trainer.__init__).parameters.keys():
        if param not in skip_params:
            assert f'--trainer.{param}' in out.getvalue()

    cli_args = ['any.py', '--data.help=tests.helpers.BoringDataModule']
    out = StringIO()
    with mock.patch('sys.argv', cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    assert '--data.init_args.data_dir' in out.getvalue()


def test_lightning_cli_print_config():

    cli_args = [
        'any.py',
        '--seed_everything=1234',
        '--model=tests.helpers.BoringModel',
        '--data=tests.helpers.BoringDataModule',
        '--print_config',
    ]

    out = StringIO()
    with mock.patch('sys.argv', cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    outval = yaml.safe_load(out.getvalue())
    assert outval['seed_everything'] == 1234
    assert outval['model']['class_path'] == 'tests.helpers.BoringModel'
    assert outval['data']['class_path'] == 'tests.helpers.BoringDataModule'


def test_lightning_cli_submodules(tmpdir):

    class MainModule(BoringModel):

        def __init__(
            self,
            submodule1: LightningModule,
            submodule2: LightningModule,
            main_param: int = 1,
        ):
            super().__init__()
            self.submodule1 = submodule1
            self.submodule2 = submodule2

    config = """model:
        main_param: 2
        submodule1:
            class_path: tests.helpers.BoringModel
        submodule2:
            class_path: tests.helpers.BoringModel
    """
    config_path = tmpdir / 'config.yaml'
    with open(config_path, 'w') as f:
        f.write(config)

    cli_args = [
        f'--trainer.default_root_dir={tmpdir}',
        '--trainer.max_epochs=1',
        f'--config={str(config_path)}',
    ]

    with mock.patch('sys.argv', ['any.py'] + cli_args):
        cli = LightningCLI(MainModule)

    assert cli.config_init['model']['main_param'] == 2
    assert cli.model.submodule1 == cli.config_init['model']['submodule1']
    assert cli.model.submodule2 == cli.config_init['model']['submodule2']
    assert isinstance(cli.config_init['model']['submodule1'], BoringModel)
    assert isinstance(cli.config_init['model']['submodule2'], BoringModel)
