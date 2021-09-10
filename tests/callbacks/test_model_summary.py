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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import ModelSummaryMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_model_summary_callback_present_trainer():

    trainer = Trainer()
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    trainer = Trainer(callbacks=ModelSummary())
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)


def test_model_summary_callback_with_weights_summary_none():

    trainer = Trainer(weights_summary=None)
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)


def test_model_summary_callback_with_weights_summary():

    trainer = Trainer(weights_summary="top")

    model_summary_callback = list(filter(lambda cb: isinstance(cb, ModelSummary), trainer.callbacks))[0]
    assert model_summary_callback._max_depth == 1

    trainer = Trainer(weights_summary="full")

    model_summary_callback = list(filter(lambda cb: isinstance(cb, ModelSummary), trainer.callbacks))[0]
    assert model_summary_callback._max_depth == -1

    with pytest.raises(
        MisconfigurationException, match=f"`weights_summary` can be None, {', '.join(list(ModelSummaryMode))}"
    ):
        _ = Trainer(weights_summary="invalid")


def test_model_summary_callback_override_weights_summary_flag():

    trainer = Trainer(callbacks=ModelSummary(), weights_summary=None)
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)
