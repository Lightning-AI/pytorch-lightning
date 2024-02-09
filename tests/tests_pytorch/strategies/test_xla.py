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
from unittest.mock import Mock

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import XLAAccelerator
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import XLAStrategy

from tests_pytorch.helpers.runif import RunIf


class BoringModelTPU(BoringModel):
    def on_train_start(self) -> None:
        # assert strategy attributes for device setting
        assert self.device == torch.device("xla", index=0)
        assert os.environ.get("PT_XLA_DEBUG") == "1"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_xla_strategy_debug_state():
    """Tests if device/debug flag is set correctly when training and after teardown for XLAStrategy."""
    model = BoringModelTPU()
    trainer = Trainer(fast_dev_run=True, strategy=XLAStrategy(debug=True))
    assert isinstance(trainer.accelerator, XLAAccelerator)
    assert isinstance(trainer.strategy, XLAStrategy)
    trainer.fit(model)
    assert "PT_XLA_DEBUG" not in os.environ


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_rank_properties_access(xla_available):
    """Test that the strategy returns the expected values depending on whether we're in the main process or not."""
    strategy = XLAStrategy()
    strategy.cluster_environment = Mock()

    # we're in the main process, no processes have been launched yet
    assert not strategy._launched
    assert strategy.global_rank == 0
    assert strategy.local_rank == 0
    assert strategy.node_rank == 0
    assert strategy.world_size == 1

    # simulate we're in a worker process
    strategy._launched = True
    assert strategy.global_rank == strategy.cluster_environment.global_rank()
    assert strategy.local_rank == strategy.cluster_environment.local_rank()
    assert strategy.node_rank == strategy.cluster_environment.node_rank()
    assert strategy.world_size == strategy.cluster_environment.world_size()
