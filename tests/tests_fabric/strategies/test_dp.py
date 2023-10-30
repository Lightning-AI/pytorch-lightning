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
from lightning.fabric.strategies import DataParallelStrategy

from tests_fabric.helpers.runif import RunIf
from tests_fabric.strategies.test_single_device import _MyFabricGradNorm, _MyFabricGradVal


def test_data_parallel_root_device():
    strategy = DataParallelStrategy()
    strategy.parallel_devices = [torch.device("cuda", 2), torch.device("cuda", 0), torch.device("cuda", 1)]
    assert strategy.root_device == torch.device("cuda", 2)


def test_data_parallel_ranks():
    strategy = DataParallelStrategy()
    assert strategy.world_size == 1
    assert strategy.local_rank == 0
    assert strategy.global_rank == 0
    assert strategy.is_global_zero


@mock.patch("lightning.fabric.strategies.dp.DataParallel")
def test_data_parallel_setup_module(data_parallel_mock):
    strategy = DataParallelStrategy()
    strategy.parallel_devices = [0, 2, 1]
    module = torch.nn.Linear(2, 2)
    wrapped_module = strategy.setup_module(module)
    assert wrapped_module == data_parallel_mock(module=module, device_ids=[0, 2, 1])


def test_data_parallel_module_to_device():
    strategy = DataParallelStrategy()
    strategy.parallel_devices = [torch.device("cuda", 2)]
    module = Mock()
    strategy.module_to_device(module)
    module.to.assert_called_with(torch.device("cuda", 2))


def test_dp_module_state_dict():
    """Test that the module state dict gets retrieved without the prefixed wrapper keys from DP."""

    class DataParallelMock(MagicMock):
        def __instancecheck__(self, instance):
            # to make the strategy's `isinstance(model, DataParallel)` pass with a mock as class
            return True

    strategy = DataParallelStrategy(parallel_devices=[torch.device("cpu"), torch.device("cpu")])

    # Without DP applied (no setup call)
    original_module = torch.nn.Linear(2, 3)
    assert strategy.get_module_state_dict(original_module).keys() == original_module.state_dict().keys()

    # With DP applied (setup called)
    with mock.patch("lightning.fabric.strategies.dp.DataParallel", DataParallelMock):
        wrapped_module = strategy.setup_module(original_module)
        assert strategy.get_module_state_dict(wrapped_module).keys() == original_module.state_dict().keys()


@pytest.mark.parametrize(
    "precision",
    [
        "32-true",
        "16-mixed",
        pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True)),
    ],
)
@pytest.mark.parametrize("clip_type", ["norm", "val"])
@RunIf(min_cuda_gpus=2)
def test_dp_grad_clipping(clip_type, precision):
    clipping_test_cls = _MyFabricGradNorm if clip_type == "norm" else _MyFabricGradVal
    fabric = clipping_test_cls(accelerator="cuda", devices=2, precision=precision, strategy="dp")
    fabric.run()
