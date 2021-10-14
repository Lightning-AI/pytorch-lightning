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
import torch

from pytorch_lightning.trainer import Trainer
from tests.helpers import BoringModel


@pytest.mark.parametrize("gpus", [-1, "-1"])
def test_all_gpus(tmpdir, gpus):
    """Testing that the -1 is stable for GPU machines also if GPU is missing."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=gpus,
    )
    trainer.fit(model)
    assert trainer.accelerator_connector.use_gpu == torch.cuda.is_available()
    assert trainer.accelerator_connector.num_gpus == torch.cuda.device_count()
