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
"""Test deprecated functionality which will be removed in vX.Y.Z"""

import pytest
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_tbd_remove_in_v1_2_0():
    with pytest.deprecated_call(match='will be removed in v1.2'):
        ModelCheckpoint(filepath='..')

    with pytest.deprecated_call(match='will be removed in v1.2'):
        ModelCheckpoint('..')

    with pytest.raises(MisconfigurationException, match='inputs which are not feasible'):
        ModelCheckpoint(filepath='..', dirpath='.')


def test_tbd_remove_in_v1_2_0_metrics():
    from pytorch_lightning.metrics.classification import Fbeta
    from pytorch_lightning.metrics.functional.classification import f1_score, fbeta_score

    with pytest.deprecated_call(match='will be removed in v1.2'):
        Fbeta(2)

    with pytest.deprecated_call(match='will be removed in v1.2'):
        fbeta_score(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 1]), 0.2)

    with pytest.deprecated_call(match='will be removed in v1.2'):
        f1_score(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 0]))
