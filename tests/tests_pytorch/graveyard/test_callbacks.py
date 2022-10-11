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

from pytorch_lightning.callbacks import ModelCheckpoint


def test_v2_0_0_deprecated_mc_save_checkpoint():
    mc = ModelCheckpoint()
    with pytest.raises(
        NotImplementedError,
        match=r"ModelCheckpoint.save_checkpoint\(\)` was deprecated in v1.6 and is no longer supported as of 1.8.",
    ):
        mc.save_checkpoint(None)
