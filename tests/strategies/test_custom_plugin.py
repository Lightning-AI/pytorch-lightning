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
import os
from typing import Any, Mapping

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import SingleDeviceStrategy
from tests.helpers import BoringModel


@pytest.mark.parametrize("restore_optimizer_and_schedulers", [True, False])
def test_strategy_lightning_restore_optimizer_and_schedulers(tmpdir, restore_optimizer_and_schedulers):
    class TestStrategy(SingleDeviceStrategy):
        load_optimizer_state_dict_called = False

        @property
        def lightning_restore_optimizer(self) -> bool:
            return restore_optimizer_and_schedulers

        def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
            self.load_optimizer_state_dict_called = True

    # create ckpt to resume from
    checkpoint_path = os.path.join(tmpdir, "model.ckpt")
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()
    strategy = TestStrategy(torch.device("cpu"))
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy=strategy)
    trainer.fit(model, ckpt_path=checkpoint_path)
    assert strategy.load_optimizer_state_dict_called == restore_optimizer_and_schedulers
