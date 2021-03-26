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

from pytorch_lightning import Trainer


def test_passing_env_variables(tmpdir):
    """Testing overwriting trainer arguments """
    trainer = Trainer()
    assert trainer.logger is not None
    assert trainer.max_steps is None
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is None
    assert trainer.max_steps == 42

    os.environ['PL_TRAINER_LOGGER'] = 'False'
    os.environ['PL_TRAINER_MAX_STEPS'] = '7'
    trainer = Trainer()
    assert trainer.logger is None
    assert trainer.max_steps == 7

    os.environ['PL_TRAINER_LOGGER'] = 'True'
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is not None
    assert trainer.max_steps == 7

    # this has to be cleaned
    del os.environ['PL_TRAINER_LOGGER']
    del os.environ['PL_TRAINER_MAX_STEPS']
