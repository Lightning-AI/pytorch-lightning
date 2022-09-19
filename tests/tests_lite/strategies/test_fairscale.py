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

from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies import DDPShardedStrategy
from lightning_lite.strategies.fairscale import ShardedDataParallel


@RunIf(fairscale=True)
def test_block_backward_sync():
    strategy = DDPShardedStrategy()
    model = mock.MagicMock(spec=ShardedDataParallel)
    with strategy.block_backward_sync(model):
        pass
    model.no_sync.assert_called_once()
