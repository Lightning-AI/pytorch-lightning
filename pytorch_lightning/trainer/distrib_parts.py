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

"""
Root module for all distributed operations in Lightning.
Currently supports training on CPU, GPU (dp, ddp, ddp2, horovod) and TPU.

"""

from abc import ABC, abstractmethod
import torch
from typing import Union, Callable, Any, List, Optional, Tuple, MutableSequence

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from pytorch_lightning.utilities import move_data_to_device, AMPType

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class TrainerDPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_dp: bool
    use_ddp2: bool
    use_ddp: bool
    testing: bool
    use_single_gpu: bool
    root_gpu: ...
    amp_level: str
    precision: ...
    global_rank: int
    local_rank: int
    tpu_local_core_rank: int
    tpu_global_core_rank: int
    use_tpu: bool
    data_parallel_device_ids: ...
    progress_bar_callback: ...
    on_colab_kaggle: str
    save_spawn_weights: Callable
    logger: ...
    amp_backend: AMPType

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.logger = self.logger
            m.use_dp = self.use_dp
            m.use_ddp2 = self.use_ddp2
            m.use_ddp = self.use_ddp
            m.use_amp = self.amp_backend is not None
            m.testing = self.testing
            m.use_single_gpu = self.use_single_gpu
            m.use_tpu = self.use_tpu
            m.tpu_local_core_rank = self.tpu_local_core_rank
            m.tpu_global_core_rank = self.tpu_global_core_rank
            m.precision = self.precision
            m.global_rank = self.global_rank
            m.local_rank = self.local_rank

    def __transfer_batch_to_device(self, batch: Any, device: torch.device):
        model = self.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)
