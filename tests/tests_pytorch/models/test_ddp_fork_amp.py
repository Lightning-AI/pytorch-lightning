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
from torch.utils.data import DataLoader

import tests_pytorch.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import RandomDataset
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.test_models import AMPTestModel


# needs to be standalone to avoid other processes initializing CUDA
@RunIf(min_cuda_gpus_no_init=2, skip_windows=True, min_torch="1.12", standalone=True)
def test_amp_gpus_ddp_fork(tmpdir):
    """Make sure combinations of AMP and strategies work if supported."""
    tutils.reset_seed()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy="ddp_fork",
        precision=16,
    )

    model = AMPTestModel()
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model, DataLoader(RandomDataset(32, 64)))
