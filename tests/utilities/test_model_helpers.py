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
from unittest.mock import Mock

import pytest

from pl_examples.bug_report_model import BoringModel
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringDataModule


def test_is_overridden():
    model = BoringModel()
    datamodule = BoringDataModule()

    # edge cases
    assert not is_overridden("whatever", None)
    with pytest.raises(ValueError, match="Expected a parent"):
        is_overridden("whatever", object())
    assert not is_overridden("whatever", model)
    assert not is_overridden("whatever", model, parent=LightningDataModule)

    class TestModel(BoringModel):

        def foo(self):
            pass

    with pytest.raises(ValueError, match="The parent should define the method"):
        is_overridden("foo", TestModel())

    # normal usage
    assert is_overridden("training_step", model)
    assert is_overridden("train_dataloader", datamodule)

    # with mock
    mock = Mock(spec=BoringModel, wraps=model)
    assert is_overridden("training_step", mock)
    mock = Mock(spec=BoringDataModule, wraps=datamodule)
    assert is_overridden("train_dataloader", mock)
