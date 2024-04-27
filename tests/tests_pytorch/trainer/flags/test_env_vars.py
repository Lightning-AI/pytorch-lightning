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

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


def test_passing_no_env_variables():
    """Testing overwriting trainer arguments."""
    trainer = Trainer()
    model = BoringModel()
    assert trainer.logger is not None
    assert trainer.max_steps == -1
    assert trainer.max_epochs is None
    trainer = Trainer(max_steps=1, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    assert trainer.logger is None
    assert trainer.max_steps == 1
    assert trainer.max_epochs == -1


@mock.patch.dict(os.environ, {"PL_TRAINER_LOGGER": "False", "PL_TRAINER_MAX_STEPS": "7"})
def test_passing_env_variables_only():
    """Testing overwriting trainer arguments."""
    trainer = Trainer()
    assert trainer.logger is None
    assert trainer.max_steps == 7


@mock.patch.dict(os.environ, {"PL_TRAINER_LOGGER": "True", "PL_TRAINER_MAX_STEPS": "7"})
def test_passing_env_variables_defaults():
    """Testing overwriting trainer arguments."""
    trainer = Trainer(logger=False, max_steps=42)
    assert trainer.logger is None
    assert trainer.max_steps == 42


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "PL_TRAINER_DEVICES": "2"})
def test_passing_env_variables_devices(cuda_count_2, mps_count_0):
    """Testing overwriting trainer arguments."""
    trainer = Trainer()
    assert trainer.num_devices == 2
    trainer = Trainer(accelerator="gpu", devices=1)
    assert trainer.num_devices == 1
