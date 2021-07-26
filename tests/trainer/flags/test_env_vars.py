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

from pytorch_lightning import Trainer


def test_passing_no_env_variables():
    """Testing overwriting trainer arguments"""
    trainer = Trainer()
    assert trainer.logger is not None
    assert trainer.max_steps is None
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is None
    assert trainer.max_steps == 42


@mock.patch.dict(os.environ, {"PL_TRAINER_LOGGER": "False", "PL_TRAINER_MAX_STEPS": "7"})
def test_passing_env_variables_only():
    """Testing overwriting trainer arguments"""
    trainer = Trainer()
    assert trainer.logger is None
    assert trainer.max_steps == 7


@mock.patch.dict(os.environ, {"PL_TRAINER_LOGGER": "True", "PL_TRAINER_MAX_STEPS": "7"})
def test_passing_env_variables_defaults():
    """Testing overwriting trainer arguments"""
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is None
    assert trainer.max_steps == 42


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "PL_TRAINER_GPUS": "2"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_passing_env_variables_gpus(cuda_available_mock, device_count_mock):
    """Testing overwriting trainer arguments"""
    trainer = Trainer()
    assert trainer.gpus == 2
    trainer = Trainer(gpus=1)
    assert trainer.gpus == 1
