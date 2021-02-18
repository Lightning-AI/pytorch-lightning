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
from weakref import proxy


class ModelConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def copy_trainer_model_properties(self, model):
        ref_model = self.trainer.lightning_module or model

        automatic_optimization = ref_model.automatic_optimization and self.trainer.train_loop.automatic_optimization
        self.trainer.train_loop.automatic_optimization = automatic_optimization

        for m in [model, ref_model]:
            m.trainer = proxy(self.trainer)
            m._device_type = str(self.trainer._device_type)
            m._distrib_type = str(self.trainer._distrib_type)
            m.use_amp = self.trainer.amp_backend is not None
            m.testing = self.trainer.testing
            m.precision = self.trainer.precision
