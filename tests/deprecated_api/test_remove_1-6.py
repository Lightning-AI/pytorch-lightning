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
"""Test deprecated functionality which will be removed in v1.6.0."""
from unittest.mock import call, Mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringModel


def test_v1_6_0_reload_dataloaders_every_epoch(tmpdir):
    model = BoringModel()

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    with pytest.deprecated_call(match="`reload_dataloaders_every_epoch` is deprecated in v1.4 and will be removed"):
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=0.3,
            limit_val_batches=0.3,
            reload_dataloaders_every_epoch=True,
            max_epochs=3,
        )
    trainer.fit(model)
    trainer.test()

    expected_sequence = (
        [call.val_dataloader()] + [call.train_dataloader(), call.val_dataloader()] * 3 + [call.test_dataloader()]
    )
    assert tracker.mock_calls == expected_sequence


def test_v1_6_0_is_overridden_model():
    model = BoringModel()
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert is_overridden("validation_step", model=model)
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert not is_overridden("foo", model=model)


def test_v1_6_0_deprecated_disable_validation():
    trainer = Trainer()
    with pytest.deprecated_call(match="disable_validation` is deprecated in v1.4"):
        _ = trainer.disable_validation
