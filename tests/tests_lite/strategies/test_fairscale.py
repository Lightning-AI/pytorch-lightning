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

import pytest
import torch.nn as nn
import torch.optim
from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies import DDPShardedStrategy
from lightning_lite.strategies.fairscale import DDPSpawnShardedStrategy, ShardedDataParallel


@RunIf(fairscale=True)
def test_block_backward_sync():
    strategy = DDPShardedStrategy()
    model = mock.MagicMock(spec=ShardedDataParallel)
    with strategy.block_backward_sync(model):
        pass
    model.no_sync.assert_called_once()


@RunIf(fairscale=True)
@mock.patch("lightning_lite.strategies.fairscale._reinit_optimizers_with_oss", autospec=True)
@pytest.mark.parametrize("cls", [DDPShardedStrategy, DDPSpawnShardedStrategy])
def test_fairscale_custom_kwargs(_, cls):
    """Test that if custom kwargs are passed, they are set correctly."""
    strategy = cls(reduce_fp16=True)
    assert strategy._ddp_kwargs["reduce_fp16"] is True

    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with mock.patch("lightning_lite.strategies.fairscale.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.setup_module_and_optimizers(model, [optimizer])
    args, kwargs = mock_sharded.call_args
    assert kwargs["reduce_fp16"] is True


@RunIf(fairscale=True)
@mock.patch("lightning_lite.strategies.fairscale._reinit_optimizers_with_oss", autospec=True)
@pytest.mark.parametrize("kwargs, expected_buffer_size", [(dict(), 0), (dict(reduce_buffer_size=128), 128)])
@pytest.mark.parametrize("num_nodes", [1, 2])
def test_fairscale_custom_kwargs_reduce_buffer_size(_, kwargs, expected_buffer_size, num_nodes):
    """Test that ``reduce_buffer_size`` is correctly set based on provided kwargs."""
    strategy = DDPShardedStrategy(**kwargs)
    strategy.num_nodes = num_nodes

    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with mock.patch("lightning_lite.strategies.fairscale.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.setup_module_and_optimizers(model, [optimizer])

    args, kwargs = mock_sharded.call_args
    assert "reduce_buffer_size" in kwargs

    if num_nodes > 1 and len(kwargs) == 0:
        # If user has not specified a buffer size, and we're using multiple nodes, check to see if default is set
        assert kwargs["reduce_buffer_size"] == DDPShardedStrategy._REDUCE_BUFFER_SIZE_DEFAULT
    else:
        assert kwargs["reduce_buffer_size"] == expected_buffer_size
