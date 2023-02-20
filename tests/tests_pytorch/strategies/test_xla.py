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
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import XLAStrategy
from tests_pytorch.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests_pytorch.helpers.runif import RunIf


def test_error_process_iterable_dataloader(xla_available):
    strategy = XLAStrategy(MagicMock())
    loader_no_len = CustomNotImplementedErrorDataloader(DataLoader(RandomDataset(32, 64)))
    with pytest.raises(TypeError, match="TPUs do not currently support"):
        strategy.process_dataloader(loader_no_len)


class BoringModelTPU(BoringModel):
    def on_train_start(self) -> None:
        # assert strategy attributes for device setting
        assert self.device == torch.device("xla", index=1)
        assert os.environ.get("PT_XLA_DEBUG") == "1"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_one_core():
    """Tests if device/debug flag is set correctly when training and after teardown for XLAStrategy."""
    model = BoringModelTPU()
    trainer = Trainer(accelerator="tpu", devices=1, fast_dev_run=True, strategy=XLAStrategy(debug=True))
    assert isinstance(trainer.strategy, XLAStrategy)
    trainer.fit(model)
    assert "PT_XLA_DEBUG" not in os.environ
