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
from datetime import timedelta
from unittest import mock

import pytest
import torch
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins import DoublePrecision, HalfPrecision, Precision
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.states import TrainerFn
from torch.nn.parallel import DistributedDataParallel

from tests_pytorch.helpers.runif import RunIf


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
        ("ddp_spawn", {}),
        pytest.param("ddp_fork", {}, marks=RunIf(skip_windows=True)),
        pytest.param("ddp_notebook", {}, marks=RunIf(skip_windows=True)),
        ("ddp_spawn_find_unused_parameters_false", {"find_unused_parameters": False}),
        ("ddp_spawn_find_unused_parameters_true", {"find_unused_parameters": True}),
        pytest.param(
            "ddp_fork_find_unused_parameters_false", {"find_unused_parameters": False}, marks=RunIf(skip_windows=True)
        ),
        pytest.param(
            "ddp_fork_find_unused_parameters_true", {"find_unused_parameters": True}, marks=RunIf(skip_windows=True)
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_false",
            {"find_unused_parameters": False},
            marks=RunIf(skip_windows=True),
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_true",
            {"find_unused_parameters": True},
            marks=RunIf(skip_windows=True),
        ),
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
        (Precision(), torch.float32),
        (DoublePrecision(), torch.float64),
        (HalfPrecision("16-true"), torch.float16),
        pytest.param(HalfPrecision("bf16-true"), torch.bfloat16, marks=RunIf(bf16_cuda=True)),
    ],
)
@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_tensor_init_context(precision_plugin, expected_dtype):
    """Test that the module under the init-context gets moved to the right device and dtype."""
    parallel_devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    expected_device = parallel_devices[1]

    strategy = DDPStrategy(
        parallel_devices=parallel_devices, precision_plugin=precision_plugin, cluster_environment=LightningEnvironment()
    )
    assert strategy.local_rank == 1
    with strategy.tensor_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device == module.bias.device == expected_device
    assert module.weight.dtype == module.bias.dtype == expected_dtype


@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(mock_init_process_group):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    model = BoringModel()
    ddp_strategy = DDPStrategy(timeout=test_timedelta)
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        strategy=ddp_strategy,
    )
    # test wrap the model if fitting
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()

    process_group_backend = trainer.strategy._get_process_group_backend()
    global_rank = trainer.strategy.cluster_environment.global_rank()
    world_size = trainer.strategy.cluster_environment.world_size()
    mock_init_process_group.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


@RunIf(skip_windows=True)
def test_ddp_configure_ddp(mps_count_0):
    """Tests with ddp strategy."""
    model = BoringModel()
    ddp_strategy = DDPStrategy()
    trainer = Trainer(
        max_epochs=1,
        strategy=ddp_strategy,
    )
    # test wrap the model if fitting
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    assert isinstance(trainer.model, LightningModule)
    trainer.strategy.setup(trainer)
    # in DDPStrategy configure_ddp(), model wrapped by DistributedDataParallel
    assert isinstance(trainer.model, DistributedDataParallel)

    ddp_strategy = DDPStrategy()
    trainer = Trainer(
        max_epochs=1,
        strategy=ddp_strategy,
    )
    # test do not wrap the model if TrainerFn is not fitting
    trainer.state.fn = TrainerFn.VALIDATING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    trainer.strategy.setup(trainer)
    # in DDPStrategy configure_ddp(), model are still LightningModule
    assert isinstance(trainer.model, LightningModule)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("trainer_fn", [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING])
def test_ddp_dont_configure_sync_batchnorm(trainer_fn):
    model = BoringModel()
    model.layer = torch.nn.BatchNorm1d(10)
    ddp_strategy = DDPStrategy()
    trainer = Trainer(accelerator="gpu", devices=1, strategy=ddp_strategy, sync_batchnorm=True)
    trainer.state.fn = trainer_fn
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_environment()
    assert isinstance(trainer.model, LightningModule)
    trainer.strategy.setup(trainer)
    # because TrainerFn is not FITTING, model is not configured with sync batchnorm
    assert not isinstance(trainer.strategy.model.layer, torch.nn.modules.batchnorm.SyncBatchNorm)
