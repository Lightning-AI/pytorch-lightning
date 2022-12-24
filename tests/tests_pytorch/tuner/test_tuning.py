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
from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateFinder
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_tuner_with_distributed_strategies():
    """Test that an error is raised when tuner is used with multi-device strategy."""
    trainer = Trainer(auto_scale_batch_size=True, devices=2, strategy="ddp", accelerator="cpu")
    model = BoringModel()

    with pytest.raises(MisconfigurationException, match=r"not supported with `Trainer\(strategy='ddp'\)`"):
        trainer.tune(model)


def test_tuner_with_already_configured_batch_size_finder():
    """Test that an error is raised when tuner is already configured with BatchSizeFinder."""
    trainer = Trainer(auto_scale_batch_size=True, callbacks=[BatchSizeFinder()])
    model = BoringModel()

    with pytest.raises(MisconfigurationException, match=r"Trainer is already configured with a `BatchSizeFinder`"):
        trainer.tune(model)


def test_tuner_with_already_configured_learning_rate_finder():
    """Test that an error is raised when tuner is already configured with LearningRateFinder."""
    trainer = trainer = Trainer(auto_lr_find=True, callbacks=[LearningRateFinder()])
    model = BoringModel()

    with pytest.raises(MisconfigurationException, match=r"Trainer is already configured with a `LearningRateFinder`"):
        trainer.tune(model)


def test_tuner_with_already_configured_learning_rate_finder_and_batch_size_finder():
    """Test that an error is raised when tuner are already configured with LearningRateFinder and
    BatchSizeFinder."""
    trainer = trainer = Trainer(
        auto_lr_find=True, auto_scale_batch_size=True, callbacks=[LearningRateFinder(), BatchSizeFinder()]
    )
    model = BoringModel()

    with pytest.raises(
        MisconfigurationException,
        match=r"Trainer is already configured with `LearningRateFinder` and "
        r"`BatchSizeFinder` callbacks. Please remove them if you want to use the Tuner.",
    ):
        trainer.tune(model)
