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

from pytorch_lightning import LightningModule


def test_v1_3_0_deprecated_arguments(tmpdir):
    with pytest.deprecated_call(match="The setter for self.hparams in LightningModule is deprecated"):

        class DeprecatedHparamsModel(LightningModule):

            def __init__(self, hparams):
                super().__init__()
                self.hparams = hparams

        DeprecatedHparamsModel({})
