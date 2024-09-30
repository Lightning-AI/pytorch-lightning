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
from copy import deepcopy
from datetime import timedelta
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
from lightning.fabric.plugins import DoublePrecision, HalfPrecision, Precision
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.strategies.ddp import _DDPBackwardSyncControl
from torch.nn.parallel import DistributedDataParallel

from tests_fabric.helpers.runif import RunIf


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


def test_ddp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a DDP-wrapped module."""
    strategy = DDPStrategy()
    assert isinstance(strategy._backward_sync_control, _DDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `DistributedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(Mock(), True):
        pass

    module = MagicMock(spec=DistributedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module, False):
        pass
    module.no_sync.assert_not_called()
    with strategy._backward_sync_control.no_backward_sync(module, True):
        pass
    module.no_sync.assert_called_once()


@mock.patch("lightning.fabric.strategies.ddp.DistributedDataParallel")
def test_ddp_extra_kwargs(ddp_mock):
    """Test that additional kwargs passed to the DDPStrategy get passed down to the DistributedDataParallel wrapper."""
    module = torch.nn.Linear(1, 1)
    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")])
    strategy.setup_module(module)
    ddp_mock.assert_called_with(module=module, device_ids=None)

    ddp_mock.reset_mock()

    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")], find_unused_parameters=True)
    strategy.setup_module(module)
    ddp_mock.assert_called_with(module=module, device_ids=None, find_unused_parameters=True)


def test_ddp_module_state_dict():
    """Test that the module state dict can be retrieved and loaded without the prefixed wrapper keys from DDP."""

    class DistributedDataParallelMock(MagicMock):
        def __instancecheck__(self, instance):
            # to make the strategy's `isinstance(model, DistributedDataParallel)` pass with a mock as class
            return True

    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")])

    # Without DDP applied (no setup call)
    original_module = torch.nn.Linear(2, 3)
    original_state_dict = deepcopy(original_module.state_dict())
    retrieved_state_dict = strategy.get_module_state_dict(original_module)
    assert retrieved_state_dict.keys() == original_state_dict.keys()
    strategy.load_module_state_dict(original_module, retrieved_state_dict)

    # With DDP applied (setup called)
    with mock.patch("lightning.fabric.strategies.ddp.DistributedDataParallel", DistributedDataParallelMock):
        wrapped_module = strategy.setup_module(original_module)
        retrieved_state_dict = strategy.get_module_state_dict(wrapped_module)
    assert retrieved_state_dict.keys() == original_state_dict.keys()
    strategy.load_module_state_dict(wrapped_module, retrieved_state_dict)
    strategy.load_module_state_dict(wrapped_module, original_state_dict)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        (Precision(), torch.float32),
        (HalfPrecision("16-true"), torch.float16),
        pytest.param(HalfPrecision("bf16-true"), torch.bfloat16, marks=RunIf(bf16_cuda=True)),
        (DoublePrecision(), torch.float64),
    ],
)
@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_module_init_context(precision, expected_dtype):
    """Test that the module under the init-context gets moved to the right device and dtype."""
    parallel_devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    expected_device = parallel_devices[1]

    strategy = DDPStrategy(
        parallel_devices=parallel_devices, precision=precision, cluster_environment=LightningEnvironment()
    )
    assert strategy.local_rank == 1
    with strategy.module_init_context():
        module = torch.nn.Linear(2, 2)
    assert module.weight.device == module.bias.device == expected_device
    assert module.weight.dtype == module.bias.dtype == expected_dtype


@mock.patch.dict(os.environ, {"LOCAL_RANK": "0"})
@mock.patch("lightning.fabric.strategies.ddp.DistributedDataParallel")
@mock.patch("torch.cuda.Stream")
@mock.patch("torch.cuda.stream")
def test_setup_with_cuda_stream(cuda_stream_mock, *_):
    model = torch.nn.Linear(2, 2)
    strategy = DDPStrategy(parallel_devices=[torch.device("cpu")], cluster_environment=LightningEnvironment())
    strategy.setup_module(model)
    cuda_stream_mock.assert_not_called()

    strategy = DDPStrategy(parallel_devices=[torch.device("cuda", 0)], cluster_environment=LightningEnvironment())
    strategy.setup_module(model)
    cuda_stream_mock.assert_called_once()


@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(init_process_group_mock):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    strategy = DDPStrategy(timeout=test_timedelta, parallel_devices=[torch.device("cpu")])
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()
    process_group_backend = strategy._get_process_group_backend()
    global_rank = strategy.cluster_environment.global_rank()
    world_size = strategy.cluster_environment.world_size()
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )
