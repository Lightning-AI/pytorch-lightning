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

from unittest import mock
from unittest.mock import Mock

import pytest

import pytorch_lightning.loggers.base as logger_base
import pytorch_lightning.utilities.cli as old_cli
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.profiler.advanced import AdvancedProfiler
from pytorch_lightning.profiler.base import PassThroughProfiler
from pytorch_lightning.profiler.profiler import Profiler
from pytorch_lightning.profiler.pytorch import PyTorchProfiler, RegisterRecordFunction, ScheduleWrapper
from pytorch_lightning.profiler.simple import SimpleProfiler
from pytorch_lightning.profiler.xla import XLAProfiler
from pytorch_lightning.strategies.deepspeed import LightningDeepSpeedModule
from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tests_pytorch.helpers.runif import RunIf


def test_lightning_logger_base_deprecation_warning():
    class CustomDeprecatedLogger(logger_base.LightningLoggerBase):
        def __init__(self):
            super().__init__()

        @rank_zero_only
        def log_hyperparams(self, params):
            pass

        @rank_zero_only
        def log_metrics(self, metrics, step):
            pass

        @property
        def name(self):
            pass

        @property
        def version(self):
            pass

    with pytest.deprecated_call(
        match="The `pytorch_lightning.loggers.base.LightningLoggerBase` is deprecated in v1.7"
        " and will be removed in v1.9."
    ):
        CustomDeprecatedLogger()


def test_lightning_logger_base_rank_zero_experiment_deprecation_warning():
    with pytest.deprecated_call(
        match="The `pytorch_lightning.loggers.base.rank_zero_experiment` is deprecated in v1.7"
        " and will be removed in v1.9."
    ):
        logger_base.rank_zero_experiment(None)


def test_lightning_logger_base_dummy_experiment_deprecation_warning():
    with pytest.deprecated_call(
        match="The `pytorch_lightning.loggers.base.DummyExperiment` is deprecated in v1.7 and will be removed in v1.9."
    ):
        _ = logger_base.DummyExperiment()


def test_lightning_logger_base_dummy_logger_deprecation_warning():
    with pytest.deprecated_call(
        match="The `pytorch_lightning.loggers.base.DummyLogger` is deprecated in v1.7 and will be removed in v1.9."
    ):
        _ = logger_base.DummyLogger()


def test_lightning_logger_base_merge_dicts_deprecation_warning():
    with pytest.deprecated_call(
        match="The `pytorch_lightning.loggers.base.merge_dicts` is deprecated in v1.7 and will be removed in v1.9."
    ):
        d1 = {"a": 1.7, "b": 2.0, "c": 1, "d": {"d1": 1, "d3": 3}}
        d2 = {"a": 1.1, "b": 2.2, "v": 1, "d": {"d1": 2, "d2": 3}}
        d3 = {"a": 1.1, "v": 2.3, "d": {"d3": 3, "d4": {"d5": 1}}}
        dflt_func = min
        agg_funcs = {"a": min, "v": max, "d": {"d1": sum}}
        logger_base.merge_dicts([d1, d2, d3], agg_funcs, dflt_func)


def test_old_lightningmodule_path():
    from pytorch_lightning.core.lightning import LightningModule

    with pytest.deprecated_call(
        match="pytorch_lightning.core.lightning.LightningModule has been deprecated in v1.7"
        " and will be removed in v1.9."
    ):
        LightningModule()


def test_old_loop_path():
    from pytorch_lightning.loops.base import Loop

    class MyLoop(Loop):
        def advance(self):
            ...

        def done(self):
            ...

        def reset(self):
            ...

    with pytest.deprecated_call(match="pytorch_lightning.loops.base.Loop has been deprecated in v1.7"):
        MyLoop()


def test_lightningCLI_seed_everything_default_to_None_deprecation_warning():
    with mock.patch("sys.argv", ["any.py"]), pytest.deprecated_call(
        match="Setting `LightningCLI.seed_everything_default` to `None` is deprecated in v1.7 "
        "and will be removed in v1.9. Set it to `False` instead."
    ):
        LightningCLI(LightningModule, run=False, seed_everything_default=None)


def test_old_callback_path():
    from pytorch_lightning.callbacks.base import Callback

    with pytest.deprecated_call(
        match="pytorch_lightning.callbacks.base.Callback has been deprecated in v1.7 and will be removed in v1.9."
    ):
        Callback()


def test_deprecated_dataloader_reset():
    trainer = Trainer()
    with pytest.deprecated_call(match="reset_train_val_dataloaders` has been deprecated in v1.7"):
        trainer.reset_train_val_dataloaders()


def test_lightningCLI_registries_register():
    with pytest.deprecated_call(match=old_cli._deprecate_registry_message):

        @old_cli.CALLBACK_REGISTRY
        class CustomCallback(SaveConfigCallback):
            pass


def test_lightningCLI_registries_register_automatically():
    with pytest.deprecated_call(match=old_cli._deprecate_auto_registry_message):
        with mock.patch("sys.argv", ["any.py"]):
            LightningCLI(BoringModel, run=False, auto_registry=True)


def test_lightningCLI_old_module_deprecation():
    with pytest.deprecated_call(match=r"LightningCLI.*deprecated in v1.7.*Use the equivalent class"):
        with mock.patch("sys.argv", ["any.py"]):
            old_cli.LightningCLI(BoringModel, run=False)

    with pytest.deprecated_call(match=r"SaveConfigCallback.*deprecated in v1.7.*Use the equivalent class"):
        old_cli.SaveConfigCallback(Mock(), Mock(), Mock())

    with pytest.deprecated_call(match=r"LightningArgumentParser.*deprecated in v1.7.*Use the equivalent class"):
        old_cli.LightningArgumentParser()

    with pytest.deprecated_call(match=r"instantiate_class.*deprecated in v1.7.*Use the equivalent function"):
        assert isinstance(old_cli.instantiate_class(tuple(), {"class_path": "pytorch_lightning.Trainer"}), Trainer)


def test_profiler_deprecation_warning():
    assert "Profiler` is deprecated in v1.7" in Profiler.__doc__


@pytest.mark.parametrize(
    "cls",
    [
        AdvancedProfiler,
        PassThroughProfiler,
        PyTorchProfiler,
        SimpleProfiler,
        pytest.param(XLAProfiler, marks=RunIf(tpu=True)),
    ],
)
def test_profiler_classes_deprecated_warning(cls):
    with pytest.deprecated_call(
        match=f"profiler.{cls.__name__}` is deprecated in v1.7 and will be removed in v1.9."
        f" Use .*profilers.{cls.__name__}` class instead."
    ):
        cls()


@pytest.mark.skipif(not _KINETO_AVAILABLE, reason="Requires PyTorch Profiler Kineto")
def test_pytorch_profiler_schedule_wrapper_deprecation_warning():
    with pytest.deprecated_call(match="ScheduleWrapper` is deprecated in v1.7 and will be removed in v1.9."):
        _ = ScheduleWrapper(None)


def test_pytorch_profiler_register_record_function_deprecation_warning():
    with pytest.deprecated_call(match="RegisterRecordFunction` is deprecated in v1.7 and will be removed in in v1.9."):
        _ = RegisterRecordFunction(None)


def test_gpu_accelerator_deprecation_warning():
    with pytest.deprecated_call(
        match=(
            "The `GPUAccelerator` has been renamed to `CUDAAccelerator` and will be removed in v1.9."
            + " Please use the `CUDAAccelerator` instead!"
        )
    ):
        GPUAccelerator()


def test_v1_9_0_deprecated_lightning_deepspeed_module():
    with pytest.deprecated_call(match=r"has been deprecated in v1.7.1 and will be removed in v1.9."):
        _ = LightningDeepSpeedModule(BoringModel(), 32)


def test_meta_utility_deprecations():
    import pytorch_lightning.utilities.meta as meta

    pytest.deprecated_call(meta.is_meta_init, match="is_meta_init.*removed in v1.9")
    pytest.deprecated_call(meta.init_meta, Mock(), match="init_meta.*removed in v1.9")
    pytest.deprecated_call(meta.get_all_subclasses, Mock, match="get_all_subclasses.*removed in v1.9")
    pytest.deprecated_call(meta.recursively_setattr, Mock(), "foo", 1, match="recursively_setattr.*removed in v1.9")
    pytest.deprecated_call(meta.materialize_module, Mock(), match="materialize_module.*removed in v1.9")
    with pytest.deprecated_call(match="init_meta_context.*removed in v1.9"):
        with meta.init_meta_context():
            pass
    pytest.deprecated_call(meta.is_on_meta_device, LightningModule(), match="is_on_meta_device.*removed in v1.9")
