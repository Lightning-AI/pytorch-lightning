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


def test_v2_0_0_deprecated_run_stage():
    trainer = Trainer()
    with pytest.raises(NotImplementedError,
                       match="`Trainer.run_stage` was deprecated in v1.6 and is no longer supported as of v1.8."):
        trainer.run_stage()
