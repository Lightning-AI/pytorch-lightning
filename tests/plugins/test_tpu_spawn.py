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
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.training_type import TPUSpawnPlugin
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


class BoringModelNoDataloaders(BoringModel):

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError


_loader = DataLoader(RandomDataset(32, 64))
_loader_no_len = CustomNotImplementedErrorDataloader(_loader)


@pytest.mark.parametrize(
    "train_dataloader, val_dataloaders, test_dataloaders, predict_dataloaders",
    [
        (_loader_no_len, None, None, None),
        (None, _loader_no_len, None, None),
        (None, None, _loader_no_len, None),
        (None, None, None, _loader_no_len),
        (None, [_loader, _loader_no_len], None, None),
    ],
)
@mock.patch("pytorch_lightning.plugins.training_type.tpu_spawn.xm")
def test_error_patched_iterable_dataloaders(
    _, tmpdir, train_dataloader, val_dataloaders, test_dataloaders, predict_dataloaders
):
    model = BoringModelNoDataloaders()
    connector = DataConnector(MagicMock())

    connector.attach_dataloaders(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        test_dataloaders=test_dataloaders,
        predict_dataloaders=predict_dataloaders,
    )

    with pytest.raises(MisconfigurationException, match="TPUs do not currently support"):
        TPUSpawnPlugin(MagicMock()).connect(model)


@mock.patch("pytorch_lightning.plugins.training_type.tpu_spawn.xm")
def test_error_process_iterable_dataloader(_):
    with pytest.raises(MisconfigurationException, match="TPUs do not currently support"):
        TPUSpawnPlugin(MagicMock()).process_dataloader(_loader_no_len)


class BoringModelTPU(BoringModel):

    def on_train_start(self) -> None:
        assert self.device == torch.device("xla")
        assert os.environ.get("PT_XLA_DEBUG") == "1"


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_one_core():
    """Tests if device/debug flag is set correctely when training and after teardown for TPUSpawnPlugin."""
    trainer = Trainer(tpu_cores=1, fast_dev_run=True, plugin=TPUSpawnPlugin(debug=True))
    # assert training type plugin attributes for device setting
    assert isinstance(trainer.training_type_plugin, TPUSpawnPlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert trainer.training_type_plugin.on_tpu
    assert trainer.training_type_plugin.root_device == torch.device("xla")
    model = BoringModelTPU()
    trainer.fit(model)
    assert "PT_XLA_DEBUG" not in os.environ
