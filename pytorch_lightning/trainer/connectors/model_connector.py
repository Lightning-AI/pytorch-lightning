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


class ModelConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def copy_trainer_model_properties(self, model):
        ref_model = self._get_reference_model(model)

        automatic_optimization = ref_model.automatic_optimization and self.trainer.train_loop.automatic_optimization
        self.trainer.train_loop.automatic_optimization = automatic_optimization

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
        return self._get_reference_model(self.trainer.model)

    def _get_reference_model(self, model):
        if self.trainer.accelerator_backend:
            return self.trainer.accelerator_backend.get_reference_model(model)
        return model
