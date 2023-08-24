# Copyright The Lightning AI team.
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
from unittest import mock

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

import lightning.pytorch as pl
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins import DoublePrecisionPlugin, HalfPrecisionPlugin, PrecisionPlugin
from lightning.pytorch.strategies import DDPStrategy
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_fit_only(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_test_only(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.test(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_fit_test(tmp_path):
    seed_everything(4321)
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)

    for out in result:
        assert out["test_acc"] > 0.7


@RunIf(skip_windows=True)
@mock.patch("torch.cuda.set_device")
@mock.patch("lightning.pytorch.accelerators.cuda._check_cuda_matmul_precision")
@mock.patch("lightning.pytorch.accelerators.cuda._clear_cuda_memory")
def test_ddp_torch_dist_is_available_in_setup(_, __, ___, cuda_count_1, mps_count_0, tmp_path):
    """Test to ensure torch distributed is available within the setup hook using ddp."""

    class TestModel(BoringModel):
        def setup(self, stage: str) -> None:
            assert torch.distributed.is_initialized()
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        strategy=DDPStrategy(process_group_backend="gloo"),
        accelerator="gpu",
        devices=1,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("precision", ["16-mixed", "32-true"])
def test_ddp_wrapper(tmp_path, precision):
    """Test parameters to ignore are carried over for DDP."""

    class WeirdModule(torch.nn.Module):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            return {"something": "something"}

    class CustomModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.weird_module = WeirdModule()

            # should be skip.
            self._ddp_params_and_buffers_to_ignore = ["something"]

    class CustomCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert isinstance(trainer.strategy.model, DistributedDataParallel)
            expected = ["something"]
            assert (
                trainer.strategy.model.parameters_to_ignore == set(expected) if _TORCH_GREATER_EQUAL_2_0 else expected
            )
            assert trainer.strategy.model.module._ddp_params_and_buffers_to_ignore == expected

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        precision=precision,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
        callbacks=CustomCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    ("process_group_backend", "device_str", "expected_process_group_backend"),
    [
        pytest.param("foo", "cpu", "foo"),
        pytest.param("foo", "cuda:0", "foo"),
        pytest.param(None, "cuda:0", "nccl"),
        pytest.param(None, "cpu", "gloo"),
    ],
)
def test_ddp_process_group_backend(process_group_backend, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockDDPStrategy(DDPStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockDDPStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    assert strategy._get_process_group_backend() == expected_process_group_backend


@pytest.mark.parametrize(
    ("strategy_name", "expected_ddp_kwargs"),
    [
        ("ddp", {}),
        ("ddp_find_unused_parameters_false", {"find_unused_parameters": False}),
        ("ddp_find_unused_parameters_true", {"find_unused_parameters": True}),
    ],
)
def test_ddp_kwargs_from_registry(strategy_name, expected_ddp_kwargs, mps_count_0):
    trainer = Trainer(strategy=strategy_name)
    assert trainer.strategy._ddp_kwargs == expected_ddp_kwargs


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ("precision_plugin", "expected_dtype"),
    [
        (PrecisionPlugin(), torch.float32),
        (DoublePrecisionPlugin(), torch.float64),
        (HalfPrecisionPlugin("16-true"), torch.float16),
        pytest.param(HalfPrecisionPlugin("bf16-true"), torch.bfloat16, marks=RunIf(bf16_cuda=True)),
    ],
)
@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_tensor_init_context(precision_plugin, expected_dtype):
    """Test that the module under the init-context gets moved to the right device and dtype."""
    parallel_devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    expected_device = parallel_devices[1] if _TORCH_GREATER_EQUAL_2_0 else torch.device("cpu")

    strategy = DDPStrategy(
        parallel_devices=parallel_devices, precision_plugin=precision_plugin, cluster_environment=LightningEnvironment()
    )
    assert strategy.local_rank == 1
    with strategy.tensor_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device == module.bias.device == expected_device
    assert module.weight.dtype == module.bias.dtype == expected_dtype
