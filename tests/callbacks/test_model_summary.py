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
from typing import List, Union

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import ModelSummaryMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


def test_model_summary_callback_present_trainer():

    trainer = Trainer()
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    trainer = Trainer(callbacks=ModelSummary())
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)


def test_model_summary_callback_with_weights_summary_none():
    with pytest.deprecated_call(match=r"weights_summary=None\)` is deprecated"):
        trainer = Trainer(weights_summary=None)
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    trainer = Trainer(enable_model_summary=False)
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    trainer = Trainer(enable_model_summary=False, weights_summary="full")
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    with pytest.deprecated_call(match=r"weights_summary=None\)` is deprecated"):
        trainer = Trainer(enable_model_summary=True, weights_summary=None)
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


def test_custom_model_summary_callback_summarize(tmpdir):
    class CustomModelSummary(ModelSummary):
        @staticmethod
        def summarize(
            summary_data: List[List[Union[str, List[str]]]],
            total_parameters: int,
            trainable_parameters: int,
            model_size: float,
        ) -> None:
            assert summary_data[1][0] == "Name"
            assert summary_data[1][1][0] == "layer"

            assert summary_data[2][0] == "Type"
            assert summary_data[2][1][0] == "Linear"

            assert summary_data[3][0] == "Params"
            assert total_parameters == 66
            assert trainable_parameters == 66

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=CustomModelSummary(), max_steps=1)

    trainer.fit(model)
