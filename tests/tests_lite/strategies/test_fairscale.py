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
import torch.optim
import torch.nn as nn
from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies import DDPShardedStrategy
from lightning_lite.strategies.fairscale import ShardedDataParallel, DDPSpawnShardedStrategy


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
    """Tests to ensure that if custom kwargs are passed, they are set correctly."""
    strategy = cls(reduce_fp16=True)
    assert strategy._ddp_kwargs["reduce_fp16"] is True

    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with mock.patch(f"lightning_lite.strategies.fairscale.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.setup_module_and_optimizers(model, [optimizer])
    args, kwargs = mock_sharded.call_args
    assert kwargs["reduce_fp16"] is True
