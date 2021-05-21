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

import torch

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


class BoringModelGPUTearDown(BoringModel):

    def on_train_start(self) -> None:
        assert self.device == torch.device("cuda:0")
        self.start_cuda_memory = torch.cuda.memory_allocated()

    def on_fit_end(self) -> None:
        assert self.device == torch.device("cpu")
        cuda_memory = torch.cuda.memory_allocated()
        assert cuda_memory < self.start_cuda_memory


@RunIf(skip_windows=True, min_gpus=1)
def test_single_gpu():
    """Tests if teardown correctly for single GPU plugin."""
    trainer = Trainer(gpus=1, fast_dev_run=True)
    model = BoringModelGPUTearDown()
    trainer.fit(model)


class BoringModelTPUTearDown(BoringModel):

    def on_fit_end(self) -> None:
        assert "PT_XLA_DEBUG" not in os.environ


@RunIf(tpu=True)
@pl_multi_process_test
def test_model_tpu_one_core():
    """Tests if teardown correctly for tpu plugin."""
    trainer = Trainer(tpu_cores=1, fast_dev_run=True)
    model = BoringModelTPUTearDown()
    trainer.fit(model)
