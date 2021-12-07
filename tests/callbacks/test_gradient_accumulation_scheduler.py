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
import math
from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


@pytest.mark.parametrize("accumulate_grad_batches", (1, 2, 3))
def test_trainer_accumulate_grad_batches_zero_grad(tmpdir, accumulate_grad_batches):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            enable_model_summary=False,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        assert trainer.accumulate_grad_batches == accumulate_grad_batches
        trainer.fit(model)

        assert sum(isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks) == 1
        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)


@pytest.mark.parametrize(
    ["accumulate_grad_batches", "expected_call_count"],
    [
        ({1: 2, 3: 4}, 10 + 5 + 5 + 3),
        ({0: 2, 2: 1}, 5 + 5 + 10 + 10),
    ],
)
def test_trainer_accumulate_grad_batches_dict_zero_grad(tmpdir, accumulate_grad_batches, expected_call_count):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=10,
            limit_val_batches=1,
            max_epochs=4,
            enable_model_summary=False,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        assert trainer.accumulate_grad_batches == accumulate_grad_batches.get(0, 1)
        trainer.fit(model)

        assert sum(isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks) == 1
        assert sgd_zero_grad.call_count == expected_call_count


def test_trainer_accumulate_grad_batches_with_callback(tmpdir):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=10,
            limit_val_batches=1,
            max_epochs=4,
            enable_model_summary=False,
            callbacks=[GradientAccumulationScheduler({1: 2, 3: 4})],
        )
        assert trainer.accumulate_grad_batches == 1
        trainer.fit(model)

        assert sum(isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks) == 1
        assert sgd_zero_grad.call_count == 10 + 5 + 5 + 3


@pytest.mark.parametrize(
    "scheduling",
    [
        {1: 2, -3: 4},
        {0: 2, "2": 1},
    ],
)
def test_invalid_keys_for_grad_accum_scheduler(scheduling):
    with pytest.raises(MisconfigurationException, match="Epoch should be an int"):
        _ = GradientAccumulationScheduler(scheduling=scheduling)


@pytest.mark.parametrize(
    "scheduling",
    [
        {1: 0, 3: 4},
        {0: 2, 2: "2"},
    ],
)
def test_invalid_values_for_grad_accum_scheduler(scheduling):
    with pytest.raises(MisconfigurationException, match="Accumulation factor should be an int"):
        _ = GradientAccumulationScheduler(scheduling=scheduling)
