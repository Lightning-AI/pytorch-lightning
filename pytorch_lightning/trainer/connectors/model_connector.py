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
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)


class ModelConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self.trainer
            m.logger = self.trainer.logger
            m.use_dp = self.trainer.use_dp
            m.use_ddp2 = self.trainer.use_ddp2
            m.use_ddp = self.trainer.use_ddp
            m.use_amp = self.trainer.amp_backend is not None
            m.testing = self.trainer.testing
            m.use_single_gpu = self.trainer.use_single_gpu
            m.use_tpu = self.trainer.use_tpu
            m.tpu_local_core_rank = self.trainer.tpu_local_core_rank
            m.tpu_global_core_rank = self.trainer.tpu_global_core_rank
            m.precision = self.trainer.precision
            m.global_rank = self.trainer.global_rank
            m.local_rank = self.trainer.local_rank

    def get_model(self):
        is_dp_module = isinstance(self.trainer.model, (LightningDistributedDataParallel, LightningDataParallel))
        model = self.trainer.model.module if is_dp_module else self.trainer.model
        return model
