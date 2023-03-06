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
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.strategies.ddp import _DDPBackwardSyncControl
from tests_fabric.helpers.runif import RunIf
from tests_fabric.strategies.test_single_device import _MyFabricGradNorm, _MyFabricGradVal


@pytest.mark.parametrize(
    ["process_group_backend", "device_str", "expected_process_group_backend"],
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
    ):
        with strategy._backward_sync_control.no_backward_sync(Mock()):
            pass

    module = MagicMock(spec=DistributedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@mock.patch("lightning.fabric.strategies.ddp.DistributedDataParallel")
def test_ddp_extra_kwargs(ddp_mock):
    """Test that additional kwargs passed to the DDPStrategy get passed down to the DistributedDataParallel
    wrapper."""
    module = torch.nn.Linear(1, 1)
    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")])
    strategy.setup_module(module)
    ddp_mock.assert_called_with(module=module, device_ids=None)

    ddp_mock.reset_mock()

    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")], find_unused_parameters=True)
    strategy.setup_module(module)
    ddp_mock.assert_called_with(module=module, device_ids=None, find_unused_parameters=True)


def test_ddp_module_state_dict():
    """Test that the module state dict gets retrieved without the prefixed wrapper keys from DDP."""

    class DistributedDataParallelMock(MagicMock):
        def __instancecheck__(self, instance):
            # to make the strategy's `isinstance(model, DistributedDataParallel)` pass with a mock as class
            return True

    strategy = DDPStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")])

    # Without DDP applied (no setup call)
    original_module = torch.nn.Linear(2, 3)
    assert strategy.get_module_state_dict(original_module).keys() == original_module.state_dict().keys()

    # With DDP applied (setup called)
    with mock.patch("lightning.fabric.strategies.ddp.DistributedDataParallel", DistributedDataParallelMock):
        wrapped_module = strategy.setup_module(original_module)
        assert strategy.get_module_state_dict(wrapped_module).keys() == original_module.state_dict().keys()


@pytest.mark.parametrize(
    "clip_type,accelerator,precision",
    [
        ("norm", "cpu", "32-true"),
        ("val", "cpu", "32-true"),
        ("norm", "cpu", "bf16-mixed"),
        ("val", "cpu", "bf16-mixed"),
        pytest.param("norm", "cuda", "32-true", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("val", "cuda", "32-true", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("norm", "cuda", "16-mixed", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("val", "cuda", "16-mixed", marks=RunIf(min_cuda_gpus=2)),
        pytest.param("norm", "cuda", "bf16-mixed", marks=RunIf(min_cuda_gpus=2, bf16_cuda=True)),
        pytest.param("val", "cuda", "bf16-mixed", marks=RunIf(min_cuda_gpus=2, bf16_cuda=True)),
    ],
)
@RunIf(standalone=True)
def test_ddp_grad_clipping(clip_type, accelerator, precision):
    if clip_type == "norm":
        clipping_test_cls = _MyFabricGradNorm
    else:
        clipping_test_cls = _MyFabricGradVal
    fabric = clipping_test_cls(accelerator=accelerator, devices=2, precision=precision, strategy="ddp")
    fabric.run()
