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
from typing import List, Optional, Union
from unittest import mock
from unittest.mock import ANY

import pytest
import torch
import yaml
from packaging import version
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.cli import (
    CALLBACK_REGISTRY,
    DATAMODULE_REGISTRY,
    instantiate_class,
    LightningArgumentParser,
    LightningCLI,
    LOGGER_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    SaveConfigCallback,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8, _TORCHVISION_AVAILABLE
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.runif import RunIf
from tests.helpers.utils import no_warning_call

torchvision_version = version.parse("0")
if _TORCHVISION_AVAILABLE:
    torchvision_version = version.parse(__import__("torchvision").__version__)


@mock.patch("argparse.ArgumentParser.parse_args")
def test_default_args(mock_argparse):
    """Tests default argument parser for Trainer."""
    mock_argparse.return_value = Namespace(**Trainer.default_attributes())

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    args = parser.parse_args([])

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5


@pytest.mark.parametrize("cli_args", [["--accumulate_grad_batches=22"], []])
def test_add_argparse_args_redefined(cli_args):
    """Redefines some default Trainer arguments via the cli and tests the Trainer initialization correctness."""
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)

    args = parser.parse_args(cli_args)

    # make sure we can pickle args
    pickle.dumps(args)

    # Check few deprecated args are not in namespace:
    for depr_name in ("gradient_clip", "nb_gpu_nodes", "max_nb_epochs"):
        assert depr_name not in args

    trainer = Trainer.from_argparse_args(args=args)
    pickle.dumps(trainer)

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize("cli_args", [["--callbacks=1", "--logger"], ["--foo", "--bar=1"]])
def test_add_argparse_args_redefined_error(cli_args, monkeypatch):
    """Asserts error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)

    monkeypatch.setattr(parser, "exit", lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)


@pytest.mark.parametrize(
    ["cli_args", "expected"],
    [
        ("--auto_lr_find=True --auto_scale_batch_size=power", dict(auto_lr_find=True, auto_scale_batch_size="power")),
        (
            "--auto_lr_find any_string --auto_scale_batch_size ON",
            dict(auto_lr_find="any_string", auto_scale_batch_size=True),
        ),
        ("--auto_lr_find=Yes --auto_scale_batch_size=On", dict(auto_lr_find=True, auto_scale_batch_size=True)),
        ("--auto_lr_find Off --auto_scale_batch_size No", dict(auto_lr_find=False, auto_scale_batch_size=False)),
        ("--auto_lr_find TRUE --auto_scale_batch_size FALSE", dict(auto_lr_find=True, auto_scale_batch_size=False)),
        ("--tpu_cores=8", dict(tpu_cores=8)),
        ("--tpu_cores=1,", dict(tpu_cores="1,")),
        ("--limit_train_batches=100", dict(limit_train_batches=100)),
        ("--limit_train_batches 0.8", dict(limit_train_batches=0.8)),
        ("--enable_model_summary FALSE", dict(enable_model_summary=False)),
        (
            "",
            dict(
                # These parameters are marked as Optional[...] in Trainer.__init__,
                # with None as default. They should not be changed by the argparse
                # interface.
                min_steps=None,
                accelerator=None,
                profiler=None,
            ),
        ),
    ],
)
def test_parse_args_parsing(cli_args, expected):
    """Test parsing simple types and None optionals not modified."""
    cli_args = cli_args.split(" ") if cli_args else []
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
        parser.add_lightning_class_args(Trainer, None)
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if "tpu_cores" not in expected or _TPU_AVAILABLE:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(
    ["cli_args", "expected", "instantiate"],
    [
        (["--gpus", "[0, 2]"], dict(gpus=[0, 2]), False),
        (["--tpu_cores=[1,3]"], dict(tpu_cores=[1, 3]), False),
        (['--accumulate_grad_batches={"5":3,"10":20}'], dict(accumulate_grad_batches={5: 3, 10: 20}), True),
    ],
)
def test_parse_args_parsing_complex_types(cli_args, expected, instantiate):
    """Test parsing complex types."""
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
        parser.add_lightning_class_args(Trainer, None)
        args = parser.parse_args()

    for k, v in expected.items():
        assert getattr(args, k) == v
    if instantiate:
        assert Trainer.from_argparse_args(args)


@pytest.mark.parametrize(["cli_args", "expected_gpu"], [("--gpus 1", [0]), ("--gpus 0,", [0]), ("--gpus 0,1", [0, 1])])
def test_parse_args_parsing_gpus(monkeypatch, cli_args, expected_gpu):
    """Test parsing of gpus and instantiation of Trainer."""
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    cli_args = cli_args.split(" ") if cli_args else []
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
        parser.add_lightning_class_args(Trainer, None)
        args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)
    assert trainer.data_parallel_device_ids == expected_gpu


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason="signature inspection while mocking is not working in Python < 3.7 despite autospec",
)
@pytest.mark.parametrize(
    ["cli_args", "extra_args"],
    [
        ({}, {}),
        (dict(logger=False), {}),
        (dict(logger=False), dict(logger=True)),
        (dict(logger=False), dict(enable_checkpointing=True)),
    ],
)
def test_init_from_argparse_args(cli_args, extra_args):
    unknown_args = dict(unknown_arg=0)

    # unknown args in the argparser/namespace should be ignored
    with mock.patch("pytorch_lightning.Trainer.__init__", autospec=True, return_value=None) as init:
        trainer = Trainer.from_argparse_args(Namespace(**cli_args, **unknown_args), **extra_args)
        expected = dict(cli_args)
        expected.update(extra_args)  # extra args should override any cli arg
        init.assert_called_with(trainer, **expected)

    # passing in unknown manual args should throw an error
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'unknown_arg'"):
        Trainer.from_argparse_args(Namespace(**cli_args), **extra_args, **unknown_args)


class Model(LightningModule):
    def __init__(self, model_param: int):
        super().__init__()
        self.model_param = model_param


def _model_builder(model_param: int) -> Model:
    return Model(model_param)


def _trainer_builder(
    limit_train_batches: int, fast_dev_run: bool = False, callbacks: Optional[Union[List[Callback], Callback]] = None
) -> Trainer:
    return Trainer(limit_train_batches=limit_train_batches, fast_dev_run=fast_dev_run, callbacks=callbacks)


@pytest.mark.parametrize(["trainer_class", "model_class"], [(Trainer, Model), (_trainer_builder, _model_builder)])
def test_lightning_cli(trainer_class, model_class, monkeypatch):
    """Test that LightningCLI correctly instantiates model, trainer and calls fit."""

    expected_model = dict(model_param=7)
    expected_trainer = dict(limit_train_batches=100)

    def fit(trainer, model):
        for k, v in expected_model.items():
            assert getattr(model, k) == v
        for k, v in expected_trainer.items():
            assert getattr(trainer, k) == v
        save_callback = [x for x in trainer.callbacks if isinstance(x, SaveConfigCallback)]
        assert len(save_callback) == 1
        save_callback[0].on_train_start(trainer, model)

    def on_train_start(callback, trainer, _):
        config_dump = callback.parser.dump(callback.config, skip_none=False)
        for k, v in expected_model.items():
            assert f"  {k}: {v}" in config_dump
        for k, v in expected_trainer.items():
            assert f"  {k}: {v}" in config_dump
        trainer.ran_asserts = True

    monkeypatch.setattr(Trainer, "fit", fit)
    monkeypatch.setattr(SaveConfigCallback, "on_train_start", on_train_start)

    with mock.patch("sys.argv", ["any.py", "fit", "--model.model_param=7", "--trainer.limit_train_batches=100"]):
        cli = LightningCLI(model_class, trainer_class=trainer_class, save_config_callback=SaveConfigCallback)
        assert hasattr(cli.trainer, "ran_asserts") and cli.trainer.ran_asserts


def test_lightning_cli_args_callbacks(tmpdir):

    callbacks = [
        dict(
            class_path="pytorch_lightning.callbacks.LearningRateMonitor",
            init_args=dict(logging_interval="epoch", log_momentum=True),
        ),
        dict(class_path="pytorch_lightning.callbacks.ModelCheckpoint", init_args=dict(monitor="NAME")),
    ]

    class TestModel(BoringModel):
        def on_fit_start(self):
            callback = [c for c in self.trainer.callbacks if isinstance(c, LearningRateMonitor)]
            assert len(callback) == 1
            assert callback[0].logging_interval == "epoch"
            assert callback[0].log_momentum is True
            callback = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            assert len(callback) == 1
            assert callback[0].monitor == "NAME"
            self.trainer.ran_asserts = True

    with mock.patch("sys.argv", ["any.py", "fit", f"--trainer.callbacks={json.dumps(callbacks)}"]):
        cli = LightningCLI(TestModel, trainer_defaults=dict(default_root_dir=str(tmpdir), fast_dev_run=True))

    assert cli.trainer.ran_asserts


@pytest.mark.parametrize("run", (False, True))
def test_lightning_cli_configurable_callbacks(tmpdir, run):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")

        def fit(self, **_):
            pass

    cli_args = ["fit"] if run else []
    cli_args += [f"--trainer.default_root_dir={tmpdir}", "--learning_rate_monitor.logging_interval=epoch"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel, run=run)

    callback = [c for c in cli.trainer.callbacks if isinstance(c, LearningRateMonitor)]
    assert len(callback) == 1
    assert callback[0].logging_interval == "epoch"


def test_lightning_cli_args_cluster_environments(tmpdir):
    plugins = [dict(class_path="pytorch_lightning.plugins.environments.SLURMEnvironment")]

    class TestModel(BoringModel):
        def on_fit_start(self):
            # Ensure SLURMEnvironment is set, instead of default LightningEnvironment
            assert isinstance(self.trainer._accelerator_connector.cluster_environment, SLURMEnvironment)
            self.trainer.ran_asserts = True

    with mock.patch("sys.argv", ["any.py", "fit", f"--trainer.plugins={json.dumps(plugins)}"]):
        cli = LightningCLI(TestModel, trainer_defaults=dict(default_root_dir=str(tmpdir), fast_dev_run=True))

    assert cli.trainer.ran_asserts


def test_lightning_cli_args(tmpdir):

    cli_args = [
        "fit",
        f"--data.data_dir={tmpdir}",
        f"--trainer.default_root_dir={tmpdir}",
        "--trainer.max_epochs=1",
        "--trainer.enable_model_summary=False",
        "--seed_everything=1234",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, BoringDataModule, trainer_defaults={"callbacks": [LearningRateMonitor()]})

    config_path = tmpdir / "lightning_logs" / "version_0" / "config.yaml"
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        loaded_config = yaml.safe_load(f.read())

    cli_config = cli.config["fit"].as_dict()
    assert cli_config["seed_everything"] == 1234
    assert "model" not in loaded_config and "model" not in cli_config  # no arguments to include
    assert loaded_config["data"] == cli_config["data"]
    assert loaded_config["trainer"] == cli_config["trainer"]


def test_lightning_cli_save_config_cases(tmpdir):

    config_path = tmpdir / "config.yaml"
    cli_args = ["fit", f"--trainer.default_root_dir={tmpdir}", "--trainer.logger=False", "--trainer.fast_dev_run=1"]

    # With fast_dev_run!=False config should not be saved
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        LightningCLI(BoringModel)
    assert not os.path.isfile(config_path)

    # With fast_dev_run==False config should be saved
    cli_args[-1] = "--trainer.max_epochs=1"
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        LightningCLI(BoringModel)
    assert os.path.isfile(config_path)

    # If run again on same directory exception should be raised since config file already exists
    with mock.patch("sys.argv", ["any.py"] + cli_args), pytest.raises(RuntimeError):
        LightningCLI(BoringModel)


def test_lightning_cli_config_and_subclass_mode(tmpdir):
    input_config = {
        "fit": {
            "model": {"class_path": "tests.helpers.BoringModel"},
            "data": {"class_path": "tests.helpers.BoringDataModule", "init_args": {"data_dir": str(tmpdir)}},
            "trainer": {"default_root_dir": str(tmpdir), "max_epochs": 1, "enable_model_summary": False},
        }
    }
    config_path = tmpdir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(input_config))

    with mock.patch("sys.argv", ["any.py", "--config", str(config_path)]):
        cli = LightningCLI(
            BoringModel,
            BoringDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            trainer_defaults={"callbacks": LearningRateMonitor()},
        )

    config_path = tmpdir / "lightning_logs" / "version_0" / "config.yaml"
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        loaded_config = yaml.safe_load(f.read())

    cli_config = cli.config["fit"].as_dict()
    assert loaded_config["model"] == cli_config["model"]
    assert loaded_config["data"] == cli_config["data"]
    assert loaded_config["trainer"] == cli_config["trainer"]


def any_model_any_data_cli():
    LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)


def test_lightning_cli_help():

    cli_args = ["any.py", "fit", "--help"]
    out = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()
    out = out.getvalue()

    assert "--print_config" in out
    assert "--config" in out
    assert "--seed_everything" in out
    assert "--model.help" in out
    assert "--data.help" in out

    skip_params = {"self"}
    for param in inspect.signature(Trainer.__init__).parameters.keys():
        if param not in skip_params:
            assert f"--trainer.{param}" in out

    cli_args = ["any.py", "fit", "--data.help=tests.helpers.BoringDataModule"]
    out = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    assert "--data.init_args.data_dir" in out.getvalue()


def test_lightning_cli_print_config():
    cli_args = [
        "any.py",
        "predict",
        "--seed_everything=1234",
        "--model=tests.helpers.BoringModel",
        "--data=tests.helpers.BoringDataModule",
        "--print_config",
    ]
    out = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    outval = yaml.safe_load(out.getvalue())
    assert outval["seed_everything"] == 1234
    assert outval["model"]["class_path"] == "tests.helpers.BoringModel"
    assert outval["data"]["class_path"] == "tests.helpers.BoringDataModule"
    assert outval["ckpt_path"] is None


def test_lightning_cli_submodules(tmpdir):
    class MainModule(BoringModel):
        def __init__(self, submodule1: LightningModule, submodule2: LightningModule, main_param: int = 1):
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
    config_path = tmpdir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(config)

    cli_args = [f"--trainer.default_root_dir={tmpdir}", f"--config={str(config_path)}"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(MainModule, run=False)

    assert cli.config["model"]["main_param"] == 2
    assert isinstance(cli.model.submodule1, BoringModel)
    assert isinstance(cli.model.submodule2, BoringModel)


@pytest.mark.skipif(torchvision_version < version.parse("0.8.0"), reason="torchvision>=0.8.0 is required")
def test_lightning_cli_torch_modules(tmpdir):
    class TestModule(BoringModel):
        def __init__(self, activation: torch.nn.Module = None, transform: Optional[List[torch.nn.Module]] = None):
            super().__init__()
            self.activation = activation
            self.transform = transform

    config = """model:
        activation:
          class_path: torch.nn.LeakyReLU
          init_args:
            negative_slope: 0.2
        transform:
          - class_path: torchvision.transforms.Resize
            init_args:
              size: 64
          - class_path: torchvision.transforms.CenterCrop
            init_args:
              size: 64
    """
    config_path = tmpdir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(config)

    cli_args = [f"--trainer.default_root_dir={tmpdir}", f"--config={str(config_path)}"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(TestModule, run=False)

    assert isinstance(cli.model.activation, torch.nn.LeakyReLU)
    assert cli.model.activation.negative_slope == 0.2
    assert len(cli.model.transform) == 2
    assert all(isinstance(v, torch.nn.Module) for v in cli.model.transform)


class BoringModelRequiredClasses(BoringModel):
    def __init__(self, num_classes: int, batch_size: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size


class BoringDataModuleBatchSizeAndClasses(BoringDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 5  # only available after instantiation


def test_lightning_cli_link_arguments(tmpdir):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size")
            parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")

    cli_args = [f"--trainer.default_root_dir={tmpdir}", "--data.batch_size=12"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModelRequiredClasses, BoringDataModuleBatchSizeAndClasses, run=False)

    assert cli.model.batch_size == 12
    assert cli.model.num_classes == 5

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.init_args.batch_size")
            parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")

    cli_args[-1] = "--model=tests.utilities.test_cli.BoringModelRequiredClasses"

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(
            BoringModelRequiredClasses, BoringDataModuleBatchSizeAndClasses, subclass_mode_model=True, run=False
        )

    assert cli.model.batch_size == 8
    assert cli.model.num_classes == 5


class EarlyExitTestModel(BoringModel):
    def on_fit_start(self):
        raise MisconfigurationException("Error on fit start")


@RunIf(skip_windows=True)
@pytest.mark.parametrize("logger", (False, True))
@pytest.mark.parametrize("strategy", ("ddp_spawn", "ddp"))
def test_cli_distributed_save_config_callback(tmpdir, logger, strategy):
    if _TORCH_GREATER_EQUAL_1_8:
        from torch.multiprocessing import ProcessRaisedException
    else:
        ProcessRaisedException = Exception

    with mock.patch("sys.argv", ["any.py", "fit"]), pytest.raises(
        (MisconfigurationException, ProcessRaisedException), match=r"Error on fit start"
    ):
        LightningCLI(
            EarlyExitTestModel,
            trainer_defaults={
                "default_root_dir": str(tmpdir),
                "logger": logger,
                "max_steps": 1,
                "max_epochs": 1,
                "strategy": strategy,
                "accelerator": "auto",
                "devices": 1,
            },
        )
    if logger:
        config_dir = tmpdir / "lightning_logs"
        # no more version dirs should get created
        assert os.listdir(config_dir) == ["version_0"]
        config_path = config_dir / "version_0" / "config.yaml"
    else:
        config_path = tmpdir / "config.yaml"
    assert os.path.isfile(config_path)


def test_cli_config_overwrite(tmpdir):
    trainer_defaults = {"default_root_dir": str(tmpdir), "logger": False, "max_steps": 1, "max_epochs": 1}

    argv = ["any.py", "fit"]
    with mock.patch("sys.argv", argv):
        LightningCLI(BoringModel, trainer_defaults=trainer_defaults)
    with mock.patch("sys.argv", argv), pytest.raises(RuntimeError, match="Aborting to avoid overwriting"):
        LightningCLI(BoringModel, trainer_defaults=trainer_defaults)
    with mock.patch("sys.argv", argv):
        LightningCLI(BoringModel, save_config_overwrite=True, trainer_defaults=trainer_defaults)


@pytest.mark.parametrize("run", (False, True))
def test_lightning_cli_optimizer(tmpdir, run):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)

    match = "BoringModel.configure_optimizers` will be overridden by " "`MyLightningCLI.configure_optimizers`"
    argv = ["fit", f"--trainer.default_root_dir={tmpdir}", "--trainer.fast_dev_run=1"] if run else []
    with mock.patch("sys.argv", ["any.py"] + argv), pytest.warns(UserWarning, match=match):
        cli = MyLightningCLI(BoringModel, run=run)

    assert cli.model.configure_optimizers is not BoringModel.configure_optimizers

    if not run:
        optimizer = cli.model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
    else:
        assert len(cli.trainer.optimizers) == 1
        assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
        assert len(cli.trainer.lr_scheduler_configs) == 0


def test_lightning_cli_optimizer_and_lr_scheduler(tmpdir):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

    cli_args = ["fit", f"--trainer.default_root_dir={tmpdir}", "--trainer.fast_dev_run=1", "--lr_scheduler.gamma=0.8"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel)

    assert cli.model.configure_optimizers is not BoringModel.configure_optimizers
    assert len(cli.trainer.optimizers) == 1
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
    assert len(cli.trainer.lr_scheduler_configs) == 1
    assert isinstance(cli.trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.ExponentialLR)
    assert cli.trainer.lr_scheduler_configs[0].scheduler.gamma == 0.8


def test_cli_no_need_configure_optimizers():
    class BoringModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def training_step(self, *_):
            ...

        def train_dataloader(self):
            ...

        # did not define `configure_optimizers`

    from pytorch_lightning.trainer.configuration_validator import __verify_train_val_loop_configuration

    with mock.patch("sys.argv", ["any.py", "fit", "--optimizer=Adam"]), mock.patch(
        "pytorch_lightning.Trainer._run_train"
    ) as run, mock.patch(
        "pytorch_lightning.trainer.configuration_validator.__verify_train_val_loop_configuration",
        wraps=__verify_train_val_loop_configuration,
    ) as verify:
        cli = LightningCLI(BoringModel)
    run.assert_called_once()
    verify.assert_called_once_with(cli.trainer, cli.model)


def test_lightning_cli_optimizer_and_lr_scheduler_subclasses(tmpdir):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))
            parser.add_lr_scheduler_args((torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.ExponentialLR))

    optimizer_arg = dict(class_path="torch.optim.Adam", init_args=dict(lr=0.01))
    lr_scheduler_arg = dict(class_path="torch.optim.lr_scheduler.StepLR", init_args=dict(step_size=50))
    cli_args = [
        "fit",
        f"--trainer.default_root_dir={tmpdir}",
        "--trainer.max_epochs=1",
        f"--optimizer={json.dumps(optimizer_arg)}",
        f"--lr_scheduler={json.dumps(lr_scheduler_arg)}",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel)

    assert len(cli.trainer.optimizers) == 1
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
    assert len(cli.trainer.lr_scheduler_configs) == 1
    assert isinstance(cli.trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.StepLR)
    assert cli.trainer.lr_scheduler_configs[0].scheduler.step_size == 50


@pytest.mark.parametrize("use_registries", [False, True])
def test_lightning_cli_optimizers_and_lr_scheduler_with_link_to(use_registries, tmpdir):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(
                OPTIMIZER_REGISTRY.classes if use_registries else torch.optim.Adam,
                nested_key="optim1",
                link_to="model.optim1",
            )
            parser.add_optimizer_args((torch.optim.ASGD, torch.optim.SGD), nested_key="optim2", link_to="model.optim2")
            parser.add_lr_scheduler_args(
                LR_SCHEDULER_REGISTRY.classes if use_registries else torch.optim.lr_scheduler.ExponentialLR,
                link_to="model.scheduler",
            )

    class TestModel(BoringModel):
        def __init__(self, optim1: dict, optim2: dict, scheduler: dict):
            super().__init__()
            self.optim1 = instantiate_class(self.parameters(), optim1)
            self.optim2 = instantiate_class(self.parameters(), optim2)
            self.scheduler = instantiate_class(self.optim1, scheduler)

    cli_args = ["fit", f"--trainer.default_root_dir={tmpdir}", "--trainer.max_epochs=1", "--lr_scheduler.gamma=0.2"]
    if use_registries:
        cli_args += [
            "--optim1",
            "Adam",
            "--optim1.weight_decay",
            "0.001",
            "--optim2=SGD",
            "--optim2.lr=0.01",
            "--lr_scheduler=ExponentialLR",
        ]
    else:
        cli_args += ["--optim2.class_path=torch.optim.SGD", "--optim2.init_args.lr=0.01"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(TestModel)

    assert isinstance(cli.model.optim1, torch.optim.Adam)
    assert isinstance(cli.model.optim2, torch.optim.SGD)
    assert cli.model.optim2.param_groups[0]["lr"] == 0.01
    assert isinstance(cli.model.scheduler, torch.optim.lr_scheduler.ExponentialLR)


@pytest.mark.parametrize("fn", [fn.value for fn in TrainerFn])
def test_lightning_cli_trainer_fn(fn):
    class TestCLI(LightningCLI):
        def __init__(self, *args, **kwargs):
            self.called = []
            super().__init__(*args, **kwargs)

        def before_fit(self):
            self.called.append("before_fit")

        def fit(self, **_):
            self.called.append("fit")

        def after_fit(self):
            self.called.append("after_fit")

        def before_validate(self):
            self.called.append("before_validate")

        def validate(self, **_):
            self.called.append("validate")

        def after_validate(self):
            self.called.append("after_validate")

        def before_test(self):
            self.called.append("before_test")

        def test(self, **_):
            self.called.append("test")

        def after_test(self):
            self.called.append("after_test")

        def before_predict(self):
            self.called.append("before_predict")

        def predict(self, **_):
            self.called.append("predict")

        def after_predict(self):
            self.called.append("after_predict")

        def before_tune(self):
            self.called.append("before_tune")

        def tune(self, **_):
            self.called.append("tune")

        def after_tune(self):
            self.called.append("after_tune")

    with mock.patch("sys.argv", ["any.py", fn]):
        cli = TestCLI(BoringModel)
    assert cli.called == [f"before_{fn}", fn, f"after_{fn}"]


def test_lightning_cli_subcommands():
    subcommands = LightningCLI.subcommands()
    trainer = Trainer()
    for subcommand, exclude in subcommands.items():
        fn = getattr(trainer, subcommand)
        parameters = list(inspect.signature(fn).parameters)
        for e in exclude:
            # if this fails, it's because the parameter has been removed from the associated `Trainer` function
            # and the `LightningCLI` subcommand exclusion list needs to be updated
            assert e in parameters


def test_lightning_cli_custom_subcommand():
    class TestTrainer(Trainer):
        def foo(self, model: LightningModule, x: int, y: float = 1.0):
            """Sample extra function.

            Args:
                model: A model
                x: The x
                y: The y
            """

    class TestCLI(LightningCLI):
        @staticmethod
        def subcommands():
            subcommands = LightningCLI.subcommands()
            subcommands["foo"] = {"model"}
            return subcommands

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "-h"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestCLI(BoringModel, trainer_class=TestTrainer)
    out = out.getvalue()
    assert "Sample extra function." in out
    assert "{fit,validate,test,predict,tune,foo}" in out

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "foo", "-h"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestCLI(BoringModel, trainer_class=TestTrainer)
    out = out.getvalue()
    assert "A model" not in out
    assert "Sample extra function:" in out
    assert "--x X" in out
    assert "The x (required, type: int)" in out
    assert "--y Y" in out
    assert "The y (type: float, default: 1.0)" in out


def test_lightning_cli_run():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)
    assert cli.trainer.global_step == 0
    assert isinstance(cli.trainer, Trainer)
    assert isinstance(cli.model, LightningModule)

    with mock.patch("sys.argv", ["any.py", "fit"]):
        cli = LightningCLI(BoringModel, trainer_defaults={"max_steps": 1, "max_epochs": 1})
    assert cli.trainer.global_step == 1
    assert isinstance(cli.trainer, Trainer)
    assert isinstance(cli.model, LightningModule)


@OPTIMIZER_REGISTRY
class CustomAdam(torch.optim.Adam):
    pass


@LR_SCHEDULER_REGISTRY
class CustomCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    pass


@CALLBACK_REGISTRY
class CustomCallback(Callback):
    pass


@LOGGER_REGISTRY
class CustomLogger(LightningLoggerBase):
    pass


def test_registries():
    assert "SGD" in OPTIMIZER_REGISTRY.names
    assert "RMSprop" in OPTIMIZER_REGISTRY.names
    assert "CustomAdam" in OPTIMIZER_REGISTRY.names

    assert "CosineAnnealingLR" in LR_SCHEDULER_REGISTRY.names
    assert "CosineAnnealingWarmRestarts" in LR_SCHEDULER_REGISTRY.names
    assert "CustomCosineAnnealingLR" in LR_SCHEDULER_REGISTRY.names
    assert "ReduceLROnPlateau" in LR_SCHEDULER_REGISTRY.names

    assert "EarlyStopping" in CALLBACK_REGISTRY.names
    assert "CustomCallback" in CALLBACK_REGISTRY.names

    with pytest.raises(MisconfigurationException, match="is already present in the registry"):
        OPTIMIZER_REGISTRY.register_classes(torch.optim, torch.optim.Optimizer)
    OPTIMIZER_REGISTRY.register_classes(torch.optim, torch.optim.Optimizer, override=True)

    # test `_Registry.__call__` returns the class
    assert isinstance(CustomCallback(), CustomCallback)

    assert "WandbLogger" in LOGGER_REGISTRY
    assert "CustomLogger" in LOGGER_REGISTRY


@MODEL_REGISTRY
class TestModel(BoringModel):
    def __init__(self, foo, bar=5):
        super().__init__()
        self.foo = foo
        self.bar = bar


MODEL_REGISTRY(cls=BoringModel)


def test_lightning_cli_model_choices():
    with mock.patch("sys.argv", ["any.py", "fit", "--model=BoringModel"]), mock.patch(
        "pytorch_lightning.Trainer._fit_impl"
    ) as run:
        cli = LightningCLI(trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.model, BoringModel)
        run.assert_called_once_with(cli.model, ANY, ANY, ANY, ANY)

    with mock.patch("sys.argv", ["any.py", "--model=TestModel", "--model.foo", "123"]):
        cli = LightningCLI(run=False)
        assert isinstance(cli.model, TestModel)
        assert cli.model.foo == 123
        assert cli.model.bar == 5


@DATAMODULE_REGISTRY
class MyDataModule(BoringDataModule):
    def __init__(self, foo, bar=5):
        super().__init__()
        self.foo = foo
        self.bar = bar


DATAMODULE_REGISTRY(cls=BoringDataModule)


def test_lightning_cli_datamodule_choices():
    # with set model
    with mock.patch("sys.argv", ["any.py", "fit", "--data=BoringDataModule"]), mock.patch(
        "pytorch_lightning.Trainer._fit_impl"
    ) as run:
        cli = LightningCLI(BoringModel, trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.datamodule, BoringDataModule)
        run.assert_called_once_with(ANY, ANY, ANY, cli.datamodule, ANY)

    with mock.patch("sys.argv", ["any.py", "--data=MyDataModule", "--data.foo", "123"]):
        cli = LightningCLI(BoringModel, run=False)
        assert isinstance(cli.datamodule, MyDataModule)
        assert cli.datamodule.foo == 123
        assert cli.datamodule.bar == 5

    # with configurable model
    with mock.patch("sys.argv", ["any.py", "fit", "--model", "BoringModel", "--data=BoringDataModule"]), mock.patch(
        "pytorch_lightning.Trainer._fit_impl"
    ) as run:
        cli = LightningCLI(trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.model, BoringModel)
        assert isinstance(cli.datamodule, BoringDataModule)
        run.assert_called_once_with(cli.model, ANY, ANY, cli.datamodule, ANY)

    with mock.patch("sys.argv", ["any.py", "--model", "BoringModel", "--data=MyDataModule"]):
        cli = LightningCLI(run=False)
        assert isinstance(cli.model, BoringModel)
        assert isinstance(cli.datamodule, MyDataModule)

    assert len(DATAMODULE_REGISTRY)  # needs a value initially added
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)
        # data was not passed but we are adding it automatically because there are datamodules registered
        assert "data" in cli.parser.groups
        assert not hasattr(cli.parser.groups["data"], "group_class")

    with mock.patch("sys.argv", ["any.py"]), mock.patch.dict(DATAMODULE_REGISTRY, clear=True):
        cli = LightningCLI(BoringModel, run=False)
        # no registered classes so not added automatically
        assert "data" not in cli.parser.groups
    assert len(DATAMODULE_REGISTRY)  # check state was not modified

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, BoringDataModule, run=False)
        # since we are passing the DataModule, that's whats added to the parser
        assert cli.parser.groups["data"].group_class is BoringDataModule


@pytest.mark.parametrize("use_class_path_callbacks", [False, True])
def test_registries_resolution(use_class_path_callbacks):
    """This test validates registries are used when simplified command line are being used."""
    cli_args = [
        "--optimizer",
        "Adam",
        "--optimizer.lr",
        "0.0001",
        "--trainer.callbacks=LearningRateMonitor",
        "--trainer.callbacks.logging_interval=epoch",
        "--trainer.callbacks.log_momentum=True",
        "--model=BoringModel",
        "--trainer.callbacks=ModelCheckpoint",
        "--trainer.callbacks.monitor=loss",
        "--lr_scheduler",
        "StepLR",
        "--lr_scheduler.step_size=50",
    ]

    extras = []
    if use_class_path_callbacks:
        callbacks = [
            {"class_path": "pytorch_lightning.callbacks.Callback"},
            {"class_path": "pytorch_lightning.callbacks.Callback", "init_args": {}},
        ]
        cli_args += [f"--trainer.callbacks={json.dumps(callbacks)}"]
        extras = [Callback, Callback]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(run=False)

    assert isinstance(cli.model, BoringModel)
    optimizers, lr_scheduler = cli.model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert optimizers[0].param_groups[0]["lr"] == 0.0001
    assert lr_scheduler[0].step_size == 50

    callback_types = [type(c) for c in cli.trainer.callbacks]
    expected = [LearningRateMonitor, SaveConfigCallback, ModelCheckpoint] + extras
    assert all(t in callback_types for t in expected)


def test_argv_transformation_noop():
    base = ["any.py", "--trainer.max_epochs=1"]
    argv = LightningArgumentParser._convert_argv_issue_85(CALLBACK_REGISTRY.classes, "trainer.callbacks", base)
    assert argv == base


def test_argv_transformation_single_callback():
    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + ["--trainer.callbacks=ModelCheckpoint", "--trainer.callbacks.monitor=val_loss"]
    callbacks = [
        {
            "class_path": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {"monitor": "val_loss"},
        }
    ]
    expected = base + ["--trainer.callbacks", str(callbacks)]
    argv = LightningArgumentParser._convert_argv_issue_85(CALLBACK_REGISTRY.classes, "trainer.callbacks", input)
    assert argv == expected


def test_argv_transformation_multiple_callbacks():
    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + [
        "--trainer.callbacks=ModelCheckpoint",
        "--trainer.callbacks.monitor=val_loss",
        "--trainer.callbacks=ModelCheckpoint",
        "--trainer.callbacks.monitor=val_acc",
    ]
    callbacks = [
        {
            "class_path": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {"monitor": "val_loss"},
        },
        {
            "class_path": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {"monitor": "val_acc"},
        },
    ]
    expected = base + ["--trainer.callbacks", str(callbacks)]
    argv = LightningArgumentParser._convert_argv_issue_85(CALLBACK_REGISTRY.classes, "trainer.callbacks", input)
    assert argv == expected


def test_argv_transformation_multiple_callbacks_with_config():
    base = ["any.py", "--trainer.max_epochs=1"]
    nested_key = "trainer.callbacks"
    input = base + [
        f"--{nested_key}=ModelCheckpoint",
        f"--{nested_key}.monitor=val_loss",
        f"--{nested_key}=ModelCheckpoint",
        f"--{nested_key}.monitor=val_acc",
        f"--{nested_key}=[{{'class_path': 'pytorch_lightning.callbacks.Callback'}}]",
    ]
    callbacks = [
        {
            "class_path": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {"monitor": "val_loss"},
        },
        {
            "class_path": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {"monitor": "val_acc"},
        },
        {"class_path": "pytorch_lightning.callbacks.Callback"},
    ]
    expected = base + ["--trainer.callbacks", str(callbacks)]
    nested_key = "trainer.callbacks"
    argv = LightningArgumentParser._convert_argv_issue_85(CALLBACK_REGISTRY.classes, nested_key, input)
    assert argv == expected


@pytest.mark.parametrize(
    ["args", "expected", "nested_key", "registry"],
    [
        (
            ["--optimizer", "Adadelta"],
            {"class_path": "torch.optim.adadelta.Adadelta", "init_args": {}},
            "optimizer",
            OPTIMIZER_REGISTRY,
        ),
        (
            ["--optimizer", "Adadelta", "--optimizer.lr", "10"],
            {"class_path": "torch.optim.adadelta.Adadelta", "init_args": {"lr": "10"}},
            "optimizer",
            OPTIMIZER_REGISTRY,
        ),
        (
            ["--lr_scheduler", "OneCycleLR"],
            {"class_path": "torch.optim.lr_scheduler.OneCycleLR", "init_args": {}},
            "lr_scheduler",
            LR_SCHEDULER_REGISTRY,
        ),
        (
            ["--lr_scheduler", "OneCycleLR", "--lr_scheduler.anneal_strategy=linear"],
            {"class_path": "torch.optim.lr_scheduler.OneCycleLR", "init_args": {"anneal_strategy": "linear"}},
            "lr_scheduler",
            LR_SCHEDULER_REGISTRY,
        ),
    ],
)
def test_argv_transformations_with_optimizers_and_lr_schedulers(args, expected, nested_key, registry):
    base = ["any.py", "--trainer.max_epochs=1"]
    argv = base + args
    new_argv = LightningArgumentParser._convert_argv_issue_84(registry.classes, nested_key, argv)
    assert new_argv == base + [f"--{nested_key}", str(expected)]


def test_optimizers_and_lr_schedulers_reload(tmpdir):
    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + [
        "--lr_scheduler",
        "OneCycleLR",
        "--lr_scheduler.total_steps=10",
        "--lr_scheduler.max_lr=1",
        "--optimizer",
        "Adam",
        "--optimizer.lr=0.1",
    ]

    # save config
    out = StringIO()
    with mock.patch("sys.argv", input + ["--print_config"]), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    # validate yaml
    yaml_config = out.getvalue()
    dict_config = yaml.safe_load(yaml_config)
    assert dict_config["optimizer"]["class_path"] == "torch.optim.adam.Adam"
    assert dict_config["optimizer"]["init_args"]["lr"] == 0.1
    assert dict_config["lr_scheduler"]["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"

    # reload config
    yaml_config_file = tmpdir / "config.yaml"
    yaml_config_file.write_text(yaml_config, "utf-8")
    with mock.patch("sys.argv", base + [f"--config={yaml_config_file}"]):
        LightningCLI(BoringModel, run=False)


def test_optimizers_and_lr_schedulers_add_arguments_to_parser_implemented_reload(tmpdir):
    class TestLightningCLI(LightningCLI):
        def __init__(self, *args):
            super().__init__(*args, run=False)

        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes, nested_key="opt1", link_to="model.opt1_config")
            parser.add_optimizer_args(
                (torch.optim.ASGD, torch.optim.SGD), nested_key="opt2", link_to="model.opt2_config"
            )
            parser.add_lr_scheduler_args(LR_SCHEDULER_REGISTRY.classes, link_to="model.sch_config")
            parser.add_argument("--something", type=str, nargs="+")

    class TestModel(BoringModel):
        def __init__(self, opt1_config: dict, opt2_config: dict, sch_config: dict):
            super().__init__()
            self.opt1_config = opt1_config
            self.opt2_config = opt2_config
            self.sch_config = sch_config
            opt1 = instantiate_class(self.parameters(), opt1_config)
            assert isinstance(opt1, torch.optim.Adam)
            opt2 = instantiate_class(self.parameters(), opt2_config)
            assert isinstance(opt2, torch.optim.ASGD)
            sch = instantiate_class(opt1, sch_config)
            assert isinstance(sch, torch.optim.lr_scheduler.OneCycleLR)

    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + [
        "--lr_scheduler",
        "OneCycleLR",
        "--lr_scheduler.total_steps=10",
        "--lr_scheduler.max_lr=1",
        "--opt1",
        "Adam",
        "--opt2.lr=0.1",
        "--opt2",
        "ASGD",
        "--lr_scheduler.anneal_strategy=linear",
        "--something",
        "a",
        "b",
        "c",
    ]

    # save config
    out = StringIO()
    with mock.patch("sys.argv", input + ["--print_config"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestLightningCLI(TestModel)

    # validate yaml
    yaml_config = out.getvalue()
    dict_config = yaml.safe_load(yaml_config)
    assert dict_config["opt1"]["class_path"] == "torch.optim.adam.Adam"
    assert dict_config["opt2"]["class_path"] == "torch.optim.asgd.ASGD"
    assert dict_config["opt2"]["init_args"]["lr"] == 0.1
    assert dict_config["lr_scheduler"]["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"
    assert dict_config["lr_scheduler"]["init_args"]["anneal_strategy"] == "linear"
    assert dict_config["something"] == ["a", "b", "c"]

    # reload config
    yaml_config_file = tmpdir / "config.yaml"
    yaml_config_file.write_text(yaml_config, "utf-8")
    with mock.patch("sys.argv", base + [f"--config={yaml_config_file}"]):
        cli = TestLightningCLI(TestModel)

    assert cli.model.opt1_config["class_path"] == "torch.optim.adam.Adam"
    assert cli.model.opt2_config["class_path"] == "torch.optim.asgd.ASGD"
    assert cli.model.opt2_config["init_args"]["lr"] == 0.1
    assert cli.model.sch_config["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"
    assert cli.model.sch_config["init_args"]["anneal_strategy"] == "linear"


def test_lightning_cli_config_with_subcommand():
    config = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}
    with mock.patch("sys.argv", ["any.py", f"--config={config}"]), mock.patch(
        "pytorch_lightning.Trainer.test", autospec=True
    ) as test_mock:
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1


def test_lightning_cli_config_before_subcommand():
    config = {
        "validate": {"trainer": {"limit_val_batches": 1}, "verbose": False, "ckpt_path": "barfoo"},
        "test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"},
    }

    with mock.patch("sys.argv", ["any.py", f"--config={config}", "test"]), mock.patch(
        "pytorch_lightning.Trainer.test", autospec=True
    ) as test_mock:
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1

    save_config_callback = cli.trainer.callbacks[0]
    assert save_config_callback.config.trainer.limit_test_batches == 1
    assert save_config_callback.parser.subcommand == "test"

    with mock.patch("sys.argv", ["any.py", f"--config={config}", "validate"]), mock.patch(
        "pytorch_lightning.Trainer.validate", autospec=True
    ) as validate_mock:
        cli = LightningCLI(BoringModel)

    validate_mock.assert_called_once_with(cli.trainer, cli.model, verbose=False, ckpt_path="barfoo")
    assert cli.trainer.limit_val_batches == 1

    save_config_callback = cli.trainer.callbacks[0]
    assert save_config_callback.config.trainer.limit_val_batches == 1
    assert save_config_callback.parser.subcommand == "validate"


def test_lightning_cli_config_before_subcommand_two_configs():
    config1 = {"validate": {"trainer": {"limit_val_batches": 1}, "verbose": False, "ckpt_path": "barfoo"}}
    config2 = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}

    with mock.patch("sys.argv", ["any.py", f"--config={config1}", f"--config={config2}", "test"]), mock.patch(
        "pytorch_lightning.Trainer.test", autospec=True
    ) as test_mock:
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1

    with mock.patch("sys.argv", ["any.py", f"--config={config1}", f"--config={config2}", "validate"]), mock.patch(
        "pytorch_lightning.Trainer.validate", autospec=True
    ) as validate_mock:
        cli = LightningCLI(BoringModel)

    validate_mock.assert_called_once_with(cli.trainer, cli.model, verbose=False, ckpt_path="barfoo")
    assert cli.trainer.limit_val_batches == 1


def test_lightning_cli_config_after_subcommand():
    config = {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}
    with mock.patch("sys.argv", ["any.py", "test", f"--config={config}"]), mock.patch(
        "pytorch_lightning.Trainer.test", autospec=True
    ) as test_mock:
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1


def test_lightning_cli_config_before_and_after_subcommand():
    config1 = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}
    config2 = {"trainer": {"fast_dev_run": 1}, "verbose": False, "ckpt_path": "foobar"}
    with mock.patch("sys.argv", ["any.py", f"--config={config1}", "test", f"--config={config2}"]), mock.patch(
        "pytorch_lightning.Trainer.test", autospec=True
    ) as test_mock:
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=False, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1
    assert cli.trainer.fast_dev_run == 1


def test_lightning_cli_parse_kwargs_with_subcommands(tmpdir):
    fit_config = {"trainer": {"limit_train_batches": 2}}
    fit_config_path = tmpdir / "fit.yaml"
    fit_config_path.write_text(str(fit_config), "utf8")

    validate_config = {"trainer": {"limit_val_batches": 3}}
    validate_config_path = tmpdir / "validate.yaml"
    validate_config_path.write_text(str(validate_config), "utf8")

    parser_kwargs = {
        "fit": {"default_config_files": [str(fit_config_path)]},
        "validate": {"default_config_files": [str(validate_config_path)]},
    }

    with mock.patch("sys.argv", ["any.py", "fit"]), mock.patch(
        "pytorch_lightning.Trainer.fit", autospec=True
    ) as fit_mock:
        cli = LightningCLI(BoringModel, parser_kwargs=parser_kwargs)
    fit_mock.assert_called()
    assert cli.trainer.limit_train_batches == 2
    assert cli.trainer.limit_val_batches == 1.0

    with mock.patch("sys.argv", ["any.py", "validate"]), mock.patch(
        "pytorch_lightning.Trainer.validate", autospec=True
    ) as validate_mock:
        cli = LightningCLI(BoringModel, parser_kwargs=parser_kwargs)
    validate_mock.assert_called()
    assert cli.trainer.limit_train_batches == 1.0
    assert cli.trainer.limit_val_batches == 3


def test_lightning_cli_subcommands_common_default_config_files(tmpdir):
    class Model(BoringModel):
        def __init__(self, foo: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foo = foo

    config = {"fit": {"model": {"foo": 123}}}
    config_path = tmpdir / "default.yaml"
    config_path.write_text(str(config), "utf8")
    parser_kwargs = {"default_config_files": [str(config_path)]}

    with mock.patch("sys.argv", ["any.py", "fit"]), mock.patch(
        "pytorch_lightning.Trainer.fit", autospec=True
    ) as fit_mock:
        cli = LightningCLI(Model, parser_kwargs=parser_kwargs)
    fit_mock.assert_called()
    assert cli.model.foo == 123


def test_lightning_cli_reinstantiate_trainer():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)
    assert cli.trainer.max_epochs == 1000

    class TestCallback(Callback):
        ...

    # make sure a new trainer can be easily created
    trainer = cli.instantiate_trainer(max_epochs=123, callbacks=[TestCallback()])
    # the new config is used
    assert trainer.max_epochs == 123
    assert {c.__class__ for c in trainer.callbacks} == {c.__class__ for c in cli.trainer.callbacks}.union(
        {TestCallback}
    )
    # the existing config is not updated
    assert cli.config_init["trainer"]["max_epochs"] is None


def test_cli_configure_optimizers_warning():
    match = "configure_optimizers` will be overridden by `LightningCLI"
    with mock.patch("sys.argv", ["any.py"]), no_warning_call(UserWarning, match=match):
        LightningCLI(BoringModel, run=False)
    with mock.patch("sys.argv", ["any.py", "--optimizer=Adam"]), pytest.warns(UserWarning, match=match):
        LightningCLI(BoringModel, run=False)


def test_cli_help_message():
    # full class path
    cli_args = ["any.py", "--optimizer.help=torch.optim.Adam"]
    classpath_help = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(classpath_help), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    cli_args = ["any.py", "--optimizer.help=Adam"]
    shorthand_help = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(shorthand_help), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    # the help messages should match
    assert shorthand_help.getvalue() == classpath_help.getvalue()
    # make sure it's not empty
    assert "Implements Adam" in shorthand_help.getvalue()


def test_cli_reducelronplateau():
    with mock.patch(
        "sys.argv", ["any.py", "--optimizer=Adam", "--lr_scheduler=ReduceLROnPlateau", "--lr_scheduler.monitor=foo"]
    ):
        cli = LightningCLI(BoringModel, run=False)
    config = cli.model.configure_optimizers()
    assert isinstance(config["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert config["lr_scheduler"]["scheduler"].monitor == "foo"


def test_cli_configureoptimizers_can_be_overridden():
    class MyCLI(LightningCLI):
        def __init__(self):
            super().__init__(BoringModel, run=False)

        @staticmethod
        def configure_optimizers(self, optimizer, lr_scheduler=None):
            assert isinstance(self, BoringModel)
            assert lr_scheduler is None
            return 123

    with mock.patch("sys.argv", ["any.py", "--optimizer=Adam"]):
        cli = MyCLI()
    assert cli.model.configure_optimizers() == 123

    # with no optimization config, we don't override
    with mock.patch("sys.argv", ["any.py"]):
        cli = MyCLI()
    [optimizer], [scheduler] = cli.model.configure_optimizers()
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler, StepLR)
    with mock.patch("sys.argv", ["any.py", "--lr_scheduler=StepLR"]):
        cli = MyCLI()
    [optimizer], [scheduler] = cli.model.configure_optimizers()
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler, StepLR)


def test_cli_parameter_with_lazy_instance_default():
    from jsonargparse import lazy_instance

    class TestModel(BoringModel):
        def __init__(self, activation: torch.nn.Module = lazy_instance(torch.nn.LeakyReLU, negative_slope=0.05)):
            super().__init__()
            self.activation = activation

    model = TestModel()
    assert isinstance(model.activation, torch.nn.LeakyReLU)

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False)
        assert isinstance(cli.model.activation, torch.nn.LeakyReLU)
        assert cli.model.activation.negative_slope == 0.05
        assert cli.model.activation is not model.activation


def test_cli_logger_shorthand():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False, trainer_defaults={"logger": False})
    assert cli.trainer.logger is None

    with mock.patch("sys.argv", ["any.py", "--trainer.logger=TensorBoardLogger", "--trainer.logger.save_dir=foo"]):
        cli = LightningCLI(TestModel, run=False, trainer_defaults={"logger": False})
    assert isinstance(cli.trainer.logger, TensorBoardLogger)

    with mock.patch("sys.argv", ["any.py", "--trainer.logger=False"]):
        cli = LightningCLI(TestModel, run=False)
    assert cli.trainer.logger is None
