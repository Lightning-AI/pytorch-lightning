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
import os

import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.upgrade_checkpoint import upgrade_checkpoint


@pytest.mark.parametrize(
    "old_checkpoint, new_checkpoint",
    [
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best": 0.34},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.34}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_score": 0.99},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.99}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_path": 'path'},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_path": 'path'}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "early_stop_callback_wait": 2, "early_stop_callback_patience": 4},
            {"epoch": 1, "global_step": 23, "callbacks": {EarlyStopping: {"wait_count": 2, "patience": 4}}},
        ),
    ],
)
def test_upgrade_checkpoint(tmpdir, old_checkpoint, new_checkpoint):
    filepath = os.path.join(tmpdir, "model.ckpt")
    torch.save(old_checkpoint, filepath)
    upgrade_checkpoint(filepath)
    updated_checkpoint = torch.load(filepath)
    assert updated_checkpoint == new_checkpoint
