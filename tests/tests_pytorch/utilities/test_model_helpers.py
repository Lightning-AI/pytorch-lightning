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
import pytest

from pytorch_lightning import LightningDataModule
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
from pytorch_lightning.utilities.model_helpers import is_overridden


def test_is_overridden():
    # edge cases
    assert not is_overridden("whatever", None)
    with pytest.raises(ValueError, match="Expected a parent"):
        is_overridden("whatever", object())
    model = BoringModel()
    assert not is_overridden("whatever", model)
    assert not is_overridden("whatever", model, parent=LightningDataModule)
    # normal usage
    assert is_overridden("training_step", model)
    datamodule = BoringDataModule()
    assert is_overridden("train_dataloader", datamodule)
