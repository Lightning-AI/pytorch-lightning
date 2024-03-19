# Copyright The Lightning AI team.
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
import torch.nn as nn
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.helpers.runif import RunIf


class SubSubModule(_DeviceDtypeModuleMixin):
    pass


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = SubSubModule()


class TopModule(BoringModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = SubModule()


class DeviceAssertCallback(Callback):
    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        rank = trainer.local_rank
        assert isinstance(model, TopModule)
        # index = None also means first device
        assert (model.device.index is None and rank == 0) or model.device.index == rank
        assert model.device == model.module.module.device


@RunIf(min_cuda_gpus=2)
def test_submodules_multi_gpu_ddp_spawn(tmp_path):
    model = TopModule()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=2,
        callbacks=[DeviceAssertCallback()],
        max_steps=1,
    )
    trainer.fit(model)
