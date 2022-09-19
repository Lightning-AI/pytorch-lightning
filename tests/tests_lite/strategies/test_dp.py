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

import torch

from lightning_lite.strategies import DataParallelStrategy


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


@mock.patch("lightning_lite.strategies.dp.DataParallel")
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
