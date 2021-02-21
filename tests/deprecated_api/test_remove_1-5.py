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
"""Test deprecated functionality which will be removed in v1.5.0"""
from unittest import mock

import pytest

from pytorch_lightning.loggers import WandbLogger


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_v1_5_0_wandb_unused_sync_step(tmpdir):
    with pytest.deprecated_call(match=r"v1.2.1 and will be removed in v1.5"):
        WandbLogger(sync_step=True)
