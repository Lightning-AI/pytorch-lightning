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
from pytorch_lightning.trainer.batch_size_scaling import scale_batch_size


class Tuner:

    def __init__(self, trainer):
        self.trainer = trainer

    def scale_batch_size(self,
                         model,
                         mode: str = 'power',
                         steps_per_trial: int = 3,
                         init_val: int = 2,
                         max_trials: int = 25,
                         batch_arg_name: str = 'batch_size',
                         **fit_kwargs):
        return scale_batch_size(
            self.trainer, model, mode, steps_per_trial, init_val, max_trials, batch_arg_name, **fit_kwargs
        )


