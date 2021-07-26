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
from unittest import mock

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.training_type_plugin.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(skip_windows=True, min_gpus=2, special=True)
def test_ddp_with_2_gpus():
    """Tests if device is set correctely when training and after teardown for DDPPlugin."""
    trainer = Trainer(gpus=2, accelerator="ddp", fast_dev_run=True)
    # assert training type plugin attributes for device setting
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu
    local_rank = trainer.training_type_plugin.local_rank
    assert trainer.training_type_plugin.root_device == torch.device(f"cuda:{local_rank}")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


class BarrierModel(BoringModel):
    def setup(self, stage=None):
        assert not isinstance(self.trainer.accelerator.model, DistributedDataParallel)
        self.trainer.accelerator.barrier("barrier before model is wrapped")

    def on_train_start(self):
        assert isinstance(self.trainer.accelerator.model, DistributedDataParallel)
        self.trainer.accelerator.barrier("barrier after model is wrapped")


@RunIf(min_gpus=4, special=True)
@mock.patch("torch.distributed.barrier")
def test_ddp_barrier_non_consecutive_device_ids(barrier_mock, tmpdir):
    """Test correct usage of barriers when device ids do not start at 0 or are not consecutive."""
    model = BoringModel()
    gpus = [1, 3]
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, gpus=gpus, accelerator="ddp")
    trainer.fit(model)
    barrier_mock.assert_any_call(device_ids=[gpus[trainer.local_rank]])
